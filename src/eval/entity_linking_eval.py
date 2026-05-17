import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent import AgentController, AgentResources, AgentRuntime
from configuration.config import LOG_DIR
from eval.sku_reranker.error_analysis import build_analysis_rows, write_analysis_rows
from eval.sku_reranker.reranker import RuleBasedSKUReranker, ScoredCandidate
from web.service import ChatService


ALL_BASELINES = ["exact_match", "fulltext", "hybrid", "hybrid_rerank"]
ANALYSIS_DIR = LOG_DIR / "eval" / "sku_reranker"
ANALYSIS_FILES = ["rerank_helped.jsonl", "rerank_hurt.jsonl", "rerank_still_wrong.jsonl"]


class NullTraceStore:
    def save(self, trace) -> None:
        return None

    def load(self, trace_id: str):
        return None

    def list_by_session(self, session_id: str):
        return []


def build_eval_service() -> ChatService:
    runtime = AgentRuntime(resources=AgentResources(llm_enabled=False), trace_store=NullTraceStore())
    controller = AgentController(runtime, enable_explain=False)
    service = ChatService(llm_enabled=False, controller=controller)
    service._owns_runtime = True
    return service


def load_dataset(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def dedupe_preserve_order(items: list[str]) -> list[str]:
    deduped = []
    seen = set()
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def get_candidates_for_rerank(service: ChatService, query: str, label: str = "SKU") -> dict[str, list[str]]:
    fulltext_candidates = service._search_entities(label, query, mode="fulltext", k=20)
    hybrid_candidates = service._search_entities(label, query, mode="hybrid", k=20)
    union_candidates = dedupe_preserve_order(fulltext_candidates + hybrid_candidates)
    return {
        "fulltext_candidates": fulltext_candidates,
        "hybrid_candidates": hybrid_candidates,
        "union_candidates": union_candidates,
    }


def get_hybrid_reference_record(
    *,
    service: ChatService,
    item: dict,
    top_k: int,
    hybrid_reference_cache: list[dict] | None,
    index: int,
) -> dict:
    if hybrid_reference_cache is not None:
        return hybrid_reference_cache[index]
    candidates = service._search_entities(item["label"], item["query"], mode="hybrid", k=top_k)
    return {
        "query": item["query"],
        "label": item["label"],
        "gold": item["gold"],
        "candidates": candidates,
        "top1_hit": bool(candidates and candidates[0] == item["gold"]),
        "topk_hit": item["gold"] in candidates[:top_k],
    }


def compute_metrics(records: list[dict], top_k: int = 3) -> dict:
    total = len(records)
    top1_hits = sum(1 for item in records if item["top1_hit"])
    topk_hits = sum(1 for item in records if item["topk_hit"])

    by_label = defaultdict(lambda: {"total": 0, "top1_hits": 0})
    sku_total = 0
    sku_top1_hits = 0
    sku_topk_hits = 0
    for item in records:
        stats = by_label[item["label"]]
        stats["total"] += 1
        stats["top1_hits"] += int(item["top1_hit"])
        if item["label"] == "SKU":
            sku_total += 1
            sku_top1_hits += int(item["top1_hit"])
            sku_topk_hits += int(item["topk_hit"])

    return {
        "total": total,
        "top1_accuracy": top1_hits / total if total else 0.0,
        "top_k": top_k,
        "topk_recall": topk_hits / total if total else 0.0,
        "by_label_accuracy": {
            label: stats["top1_hits"] / stats["total"] if stats["total"] else 0.0
            for label, stats in sorted(by_label.items())
        },
        "sku_subset_total": sku_total,
        "sku_subset_top1_accuracy": sku_top1_hits / sku_total if sku_total else 0.0,
        "sku_subset_topk_recall": sku_topk_hits / sku_total if sku_total else 0.0,
    }


def baseline_notes(baseline: str) -> dict:
    if baseline == "exact_match":
        return {
            "retrieval_view": "neo4j_exact_name_match",
            "ranking_signal": "no_score_exact_match",
        }
    if baseline == "fulltext":
        return {
            "retrieval_view": "neo4j_fulltext_index_query",
            "ranking_signal": "fulltext_index_score_desc",
        }
    if baseline == "hybrid":
        return {
            "retrieval_view": "neo4jvector_hybrid_search",
            "ranking_signal": "returned_order_from_similarity_search",
        }
    return {
        "scope": "sku_only",
        "sku_candidate_pool": "fulltext_top20_plus_hybrid_top20_union_dedup_preserve_order",
        "sku_ranking_method": "rule_reranker_weighted_sum",
        "non_sku_behavior": "reuse_hybrid",
        "retrieval_note": "union uses fulltext view plus current hybrid retrieval view",
    }


def evaluate_standard_baseline(service: ChatService, dataset: list[dict], baseline: str, top_k: int) -> tuple[dict, list[dict]]:
    records = []
    for item in dataset:
        candidates = service._search_entities(item["label"], item["query"], mode=baseline, k=top_k)
        records.append(
            {
                "query": item["query"],
                "label": item["label"],
                "gold": item["gold"],
                "candidates": candidates,
                "top1_hit": bool(candidates and candidates[0] == item["gold"]),
                "topk_hit": item["gold"] in candidates[:top_k],
                "top_k": top_k,
            }
        )
    metrics = compute_metrics(records, top_k=top_k)
    metrics["baseline_notes"] = baseline_notes(baseline)
    return metrics, records


def serialize_scored_candidate(item: ScoredCandidate) -> dict:
    return {
        "candidate": item.candidate,
        "score": item.score,
        "features": {
            "model_match": item.features.model_match,
            "capacity_match": item.features.capacity_match,
            "color_match": item.features.color_match,
            "brand_match": item.features.brand_match,
            "subject_match": item.features.subject_match,
            "conflict_tokens": item.features.conflict_tokens,
        },
        "original_index": item.original_index,
    }


def evaluate_hybrid_rerank_baseline(
    service: ChatService,
    dataset: list[dict],
    *,
    top_k: int,
    hybrid_reference_cache: list[dict] | None,
) -> tuple[dict, list[dict]]:
    reranker = RuleBasedSKUReranker()
    records = []
    for index, item in enumerate(dataset):
        if item["label"] != "SKU":
            hybrid_record = get_hybrid_reference_record(
                service=service,
                item=item,
                top_k=top_k,
                hybrid_reference_cache=hybrid_reference_cache,
                index=index,
            )
            records.append(
                {
                    **hybrid_record,
                    "top_k": top_k,
                }
            )
            continue

        candidate_groups = get_candidates_for_rerank(service, item["query"], label=item["label"])
        scored_candidates = reranker.score_candidates(item["query"], candidate_groups["union_candidates"])
        reranked_candidates = [entry.candidate for entry in scored_candidates]
        hybrid_record = get_hybrid_reference_record(
            service=service,
            item=item,
            top_k=top_k,
            hybrid_reference_cache=hybrid_reference_cache,
            index=index,
        )
        gold_rank_in_union = None
        for candidate_index, candidate in enumerate(candidate_groups["union_candidates"]):
            if candidate == item["gold"]:
                gold_rank_in_union = candidate_index
                break
        records.append(
            {
                "query": item["query"],
                "label": item["label"],
                "gold": item["gold"],
                "candidates": reranked_candidates,
                "top1_hit": bool(reranked_candidates and reranked_candidates[0] == item["gold"]),
                "topk_hit": item["gold"] in reranked_candidates[:top_k],
                "top_k": top_k,
                "fulltext_candidates": candidate_groups["fulltext_candidates"],
                "hybrid_candidates": candidate_groups["hybrid_candidates"],
                "union_candidates": candidate_groups["union_candidates"],
                "hybrid_top1": hybrid_record["candidates"][0] if hybrid_record["candidates"] else "",
                "hybrid_top1_hit": hybrid_record["top1_hit"],
                "hybrid_topk_hit": hybrid_record["topk_hit"],
                "rerank_top1": reranked_candidates[0] if reranked_candidates else "",
                "gold_rank_in_union": gold_rank_in_union,
                "top_candidate_features": serialize_scored_candidate(scored_candidates[0]) if scored_candidates else {},
            }
        )
    metrics = compute_metrics(records, top_k=top_k)
    metrics["baseline_notes"] = baseline_notes("hybrid_rerank")
    return metrics, records


def write_logs(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_markdown_report(path: Path, summary_payload: dict) -> None:
    lines = [
        "# Entity Linking Summary",
        "",
        "| baseline | candidate_pool | ranking_method | sku_total | sku_top1_accuracy | sku_topk_recall |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for baseline in ("fulltext", "hybrid", "hybrid_rerank"):
        if baseline not in summary_payload:
            continue
        metrics = summary_payload[baseline]["metrics"]
        notes = metrics.get("baseline_notes", {})
        candidate_pool = notes.get("sku_candidate_pool", notes.get("retrieval_view", ""))
        ranking_method = notes.get("sku_ranking_method", notes.get("ranking_signal", ""))
        lines.append(
            f"| {baseline} | {candidate_pool} | {ranking_method} | {metrics['sku_subset_total']} | {metrics['sku_subset_top1_accuracy']:.4f} | {metrics['sku_subset_topk_recall']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def cleanup_auxiliary_outputs(*, memory_suffix: str) -> None:
    markdown_path = LOG_DIR / "eval" / f"entity_linking_summary_{memory_suffix}.md"
    if markdown_path.exists():
        markdown_path.unlink()
    if ANALYSIS_DIR.exists():
        for filename in ANALYSIS_FILES:
            path = ANALYSIS_DIR / filename
            if path.exists():
                path.unlink()


def cleanup_stale_baseline_outputs(*, executed_baselines: list[str], memory_suffix: str) -> None:
    for baseline in ALL_BASELINES:
        if baseline in executed_baselines:
            continue
        path = LOG_DIR / "eval" / f"entity_linking_{baseline}_{memory_suffix}.jsonl"
        if path.exists():
            path.unlink()


def build_summary_payload(*, baselines: list[str], results: dict, auxiliary_artifacts: dict) -> dict:
    payload = {
        "executed_baselines": list(baselines),
        "auxiliary_artifacts": dict(auxiliary_artifacts),
    }
    for baseline, value in results.items():
        payload[baseline] = {
            "metrics": value["metrics"],
        }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate entity linking baselines.")
    parser.add_argument(
        "--dataset",
        default=str(ROOT_DIR / "data" / "eval" / "entity_linking.jsonl"),
    )
    parser.add_argument(
        "--baseline",
        choices=["exact_match", "fulltext", "hybrid", "hybrid_rerank", "all"],
        default="all",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--enable-session-memory", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.enable_session_memory:
        raise ValueError("Entity linking eval does not support session memory yet; use --enable-session-memory only for KGQA eval.")

    baselines = list(ALL_BASELINES) if args.baseline == "all" else [args.baseline]
    dataset = load_dataset(Path(args.dataset))
    memory_suffix = "memory_on" if args.enable_session_memory else "memory_off"
    cleanup_stale_baseline_outputs(executed_baselines=baselines, memory_suffix=memory_suffix)
    cleanup_auxiliary_outputs(memory_suffix=memory_suffix)
    service = build_eval_service()
    try:
        results = {}
        baseline_cache: dict[str, dict] = {}
        for baseline in baselines:
            if baseline == "hybrid_rerank":
                if args.baseline == "all" and "hybrid" not in baseline_cache:
                    raise RuntimeError("hybrid baseline must be executed before hybrid_rerank during --baseline all.")
                hybrid_reference_cache = baseline_cache["hybrid"]["records"] if "hybrid" in baseline_cache else None
                metrics, records = evaluate_hybrid_rerank_baseline(
                    service,
                    dataset,
                    top_k=args.top_k,
                    hybrid_reference_cache=hybrid_reference_cache,
                )
            else:
                metrics, records = evaluate_standard_baseline(service, dataset, baseline, args.top_k)
            write_logs(LOG_DIR / "eval" / f"entity_linking_{baseline}_{memory_suffix}.jsonl", records)
            results[baseline] = {"metrics": metrics, "records": records}
            baseline_cache[baseline] = {"metrics": metrics, "records": records}

        auxiliary_artifacts = {
            "markdown_generated": False,
            "reranker_analysis_generated": False,
            "reranker_reference_mode": (
                "reuse_executed_hybrid"
                if args.baseline == "all" and "hybrid_rerank" in baselines
                else "internal_hybrid_reference"
                if "hybrid_rerank" in baselines
                else "none"
            ),
        }
        if "hybrid_rerank" in baselines:
            analysis_rows = build_analysis_rows(results["hybrid_rerank"]["records"])
            write_analysis_rows(ANALYSIS_DIR, analysis_rows)
            auxiliary_artifacts["reranker_analysis_generated"] = True
        summary_payload = build_summary_payload(
            baselines=baselines,
            results=results,
            auxiliary_artifacts=auxiliary_artifacts,
        )
        if args.baseline == "all":
            write_markdown_report(LOG_DIR / "eval" / f"entity_linking_summary_{memory_suffix}.md", summary_payload)
            auxiliary_artifacts["markdown_generated"] = True
            summary_payload = build_summary_payload(
                baselines=baselines,
                results=results,
                auxiliary_artifacts=auxiliary_artifacts,
            )
        write_summary(LOG_DIR / "eval" / f"entity_linking_summary_{memory_suffix}.json", summary_payload)
        print_report(results)
    finally:
        service.close()


if __name__ == "__main__":
    main()
