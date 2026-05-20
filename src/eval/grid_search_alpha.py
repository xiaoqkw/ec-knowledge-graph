from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neo4j import GraphDatabase

from agent.runtime import resolve_local_embedding_model_name
from agent.tools.entity_link_tool import build_safe_fulltext_query
from configuration.config import EMBEDDING_MODEL_NAME, ENTITY_INDEX_CONFIG, NEO4J_CONFIG
from eval.entity_linking_eval import compute_metrics, load_dataset

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from neo4j_graphrag.retrievers.hybrid import HybridRetriever
    from tqdm.auto import tqdm
except ImportError:
    HuggingFaceEmbeddings = None
    HybridRetriever = None
    tqdm = None


class AlphaSearchService:
    def __init__(self):
        if HuggingFaceEmbeddings is None or HybridRetriever is None:
            raise ImportError("Alpha search dependencies are not installed in the current environment.")
        self.driver = GraphDatabase.driver(NEO4J_CONFIG["uri"], auth=NEO4J_CONFIG["auth"])
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=resolve_local_embedding_model_name(EMBEDDING_MODEL_NAME),
            encode_kwargs={"normalize_embeddings": True},
        )
        self.retrievers: dict[str, Any] = {}

    def close(self) -> None:
        self.driver.close()

    def _get_retriever(self, label: str) -> Any:
        if label not in self.retrievers:
            config = ENTITY_INDEX_CONFIG[label]
            self.retrievers[label] = HybridRetriever(
                self.driver,
                config["vector_index"],
                config["fulltext_index"],
                self.embedding_model,
            )
        return self.retrievers[label]

    def search_fulltext(self, *, label: str, query: str, top_k: int) -> list[str]:
        config = ENTITY_INDEX_CONFIG[label]
        records, _, _ = self.driver.execute_query(
            """
            CALL db.index.fulltext.queryNodes($index_name, $query) YIELD node, score
            RETURN node.name AS name, score
            ORDER BY score DESC
            LIMIT $limit
            """,
            {
                "index_name": config["fulltext_index"],
                "query": build_safe_fulltext_query(query),
                "limit": top_k,
            },
        )
        names: list[str] = []
        seen = set()
        for record in records:
            name = record.get("name")
            if not name or name in seen:
                continue
            seen.add(name)
            names.append(name)
        return names

    def search_linear(self, *, label: str, query: str, top_k: int, alpha: float) -> list[str]:
        if alpha <= 0.0:
            return self.search_fulltext(label=label, query=query, top_k=top_k)
        retriever = self._get_retriever(label)
        query_vector = self.embedding_model.embed_query(query)
        result = retriever.get_search_results(
            query_text=build_safe_fulltext_query(query),
            query_vector=query_vector,
            top_k=top_k,
            ranker="linear",
            alpha=alpha,
        )
        names: list[str] = []
        seen = set()
        for record in result.records:
            node = record.get("node")
            name = None if node is None else node.get("name")
            if not name or name in seen:
                continue
            seen.add(name)
            names.append(name)
        return names

    def search_naive(self, *, label: str, query: str, top_k: int) -> list[str]:
        retriever = self._get_retriever(label)
        query_vector = self.embedding_model.embed_query(query)
        result = retriever.get_search_results(
            query_text=build_safe_fulltext_query(query),
            query_vector=query_vector,
            top_k=top_k,
            ranker="naive",
        )
        names: list[str] = []
        seen = set()
        for record in result.records:
            node = record.get("node")
            name = None if node is None else node.get("name")
            if not name or name in seen:
                continue
            seen.add(name)
            names.append(name)
        return names


def _progress(iterable, *, total: int | None = None, desc: str = ""):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, disable=not sys.stderr.isatty())


def evaluate_alpha(service: AlphaSearchService, dataset: list[dict], *, top_k: int, alpha: float) -> tuple[dict, list[dict]]:
    records = []
    for item in _progress(dataset, total=len(dataset), desc=f"alpha={alpha:.2f}"):
        candidates = service.search_linear(label=item["label"], query=item["query"], top_k=top_k, alpha=alpha)
        records.append(
            {
                "query": item["query"],
                "label": item["label"],
                "gold": item["gold"],
                "candidates": candidates,
                "top1_hit": bool(candidates and candidates[0] == item["gold"]),
                "topk_hit": item["gold"] in candidates[:top_k],
                "top_k": top_k,
                "alpha": alpha,
            }
        )
    metrics = compute_metrics(records, top_k=top_k)
    metrics["baseline_notes"] = {
        "retrieval_view": "neo4j_graphrag_hybrid_linear",
        "ranking_signal": "normalized_score_linear_combination",
        "alpha": alpha,
    }
    return metrics, records


def evaluate_naive(service: AlphaSearchService, dataset: list[dict], *, top_k: int) -> tuple[dict, list[dict]]:
    records = []
    for item in _progress(dataset, total=len(dataset), desc="naive"):
        candidates = service.search_naive(label=item["label"], query=item["query"], top_k=top_k)
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
    metrics["baseline_notes"] = {
        "retrieval_view": "neo4j_graphrag_hybrid_naive",
        "ranking_signal": "normalized_score_max",
    }
    return metrics, records


def parse_alpha_values(raw: str) -> list[float]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"alpha must be between 0 and 1, got {value}")
        values.append(value)
    if not values:
        raise ValueError("No alpha values were provided.")
    return values


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search alpha for linear hybrid retrieval on an entity linking dataset.")
    parser.add_argument(
        "--dataset",
        default=str(ROOT_DIR / "data" / "eval" / "entity_linking_v2_llm_sampled_100_per_label.jsonl"),
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--alphas", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument(
        "--output",
        default=str(ROOT_DIR / "logs" / "eval" / "entity_linking_alpha_grid_search.json"),
    )
    parser.add_argument("--write-records", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(Path(args.dataset))
    alphas = parse_alpha_values(args.alphas)
    service = AlphaSearchService()
    try:
        naive_metrics, _ = evaluate_naive(service, dataset, top_k=args.top_k)
        alpha_results = []
        best_entry = None
        alpha_records: dict[str, list[dict]] = {}
        for alpha in alphas:
            metrics, records = evaluate_alpha(service, dataset, top_k=args.top_k, alpha=alpha)
            entry = {
                "alpha": alpha,
                "metrics": metrics,
            }
            alpha_results.append(entry)
            alpha_records[f"{alpha:.4f}"] = records
            if best_entry is None or metrics["top1_accuracy"] > best_entry["metrics"]["top1_accuracy"]:
                best_entry = entry

        payload = {
            "dataset": args.dataset,
            "top_k": args.top_k,
            "naive_baseline": naive_metrics,
            "best_alpha": best_entry["alpha"] if best_entry else None,
            "best_metrics": best_entry["metrics"] if best_entry else None,
            "results": alpha_results,
        }
        output_path = Path(args.output)
        write_json(output_path, payload)
        if args.write_records:
            records_dir = output_path.with_suffix("")
            for alpha_key, records in alpha_records.items():
                write_jsonl(records_dir / f"alpha_{alpha_key}.jsonl", records)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        service.close()


if __name__ == "__main__":
    main()
