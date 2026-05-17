from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def classify_help_kind(hybrid_top1_hit: bool, hybrid_topk_hit: bool, rerank_top1_hit: bool, rerank_topk_hit: bool) -> str:
    top1_fixed = not hybrid_top1_hit and rerank_top1_hit
    topk_fixed = not hybrid_topk_hit and rerank_topk_hit
    if top1_fixed and topk_fixed:
        return "top1_and_topk_fixed"
    if top1_fixed:
        return "top1_fixed"
    return "topk_fixed"


def classify_hurt_kind(hybrid_top1_hit: bool, hybrid_topk_hit: bool, rerank_top1_hit: bool, rerank_topk_hit: bool) -> str:
    top1_regression = hybrid_top1_hit and not rerank_top1_hit
    topk_regression = hybrid_topk_hit and not rerank_topk_hit
    if top1_regression and topk_regression:
        return "top1_and_topk_regression"
    if top1_regression:
        return "top1_regression"
    return "topk_regression"


def build_analysis_rows(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    helped = []
    hurt = []
    still_wrong = []
    for record in records:
        if record.get("label") != "SKU":
            continue
        hybrid_top1_hit = bool(record.get("hybrid_top1_hit", False))
        hybrid_topk_hit = bool(record.get("hybrid_topk_hit", False))
        rerank_top1_hit = bool(record.get("top1_hit", False))
        rerank_topk_hit = bool(record.get("topk_hit", False))
        payload = {
            "query": record["query"],
            "label": record["label"],
            "gold": record["gold"],
            "top_k": record["top_k"],
            "fulltext_candidates": record.get("fulltext_candidates", []),
            "hybrid_candidates": record.get("hybrid_candidates", []),
            "union_candidates": record.get("union_candidates", []),
            "reranked_candidates": record.get("candidates", []),
            "hybrid_top1": record.get("hybrid_top1", ""),
            "hybrid_top1_hit": hybrid_top1_hit,
            "hybrid_topk_hit": hybrid_topk_hit,
            "rerank_top1": record.get("rerank_top1", ""),
            "rerank_top1_hit": rerank_top1_hit,
            "rerank_topk_hit": rerank_topk_hit,
            "top_candidate_features": record.get("top_candidate_features", {}),
        }
        is_helped = ((not hybrid_top1_hit and rerank_top1_hit) or (not hybrid_topk_hit and rerank_topk_hit))
        is_hurt = ((hybrid_top1_hit and not rerank_top1_hit) or (hybrid_topk_hit and not rerank_topk_hit))
        if is_helped:
            payload["is_helped"] = True
            payload["help_kind"] = classify_help_kind(
                hybrid_top1_hit=hybrid_top1_hit,
                hybrid_topk_hit=hybrid_topk_hit,
                rerank_top1_hit=rerank_top1_hit,
                rerank_topk_hit=rerank_topk_hit,
            )
            helped.append(payload)
            continue
        if is_hurt:
            payload["is_hurt"] = True
            payload["hurt_kind"] = classify_hurt_kind(
                hybrid_top1_hit=hybrid_top1_hit,
                hybrid_topk_hit=hybrid_topk_hit,
                rerank_top1_hit=rerank_top1_hit,
                rerank_topk_hit=rerank_topk_hit,
            )
            hurt.append(payload)
            continue
        if not rerank_top1_hit:
            payload["failure_reason"] = "gold_not_in_union" if record.get("gold_rank_in_union") is None else "signal_insufficient"
            still_wrong.append(payload)
    return {
        "rerank_helped.jsonl": helped,
        "rerank_hurt.jsonl": hurt,
        "rerank_still_wrong.jsonl": still_wrong,
    }


def write_analysis_rows(output_dir: Path, grouped_rows: dict[str, list[dict[str, Any]]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in grouped_rows.items():
        path = output_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
