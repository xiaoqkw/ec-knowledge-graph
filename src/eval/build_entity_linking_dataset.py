from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve()
SRC_DIR = CURRENT_DIR.parents[1]
ROOT_DIR = CURRENT_DIR.parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(ROOT_DIR / ".env")

from configuration.config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL, ENTITY_INDEX_CONFIG, NEO4J_CONFIG
from agent.tools.entity_link_tool import build_safe_fulltext_query
from datasync.utils import Neo4jWriter
from eval.entity_linking_eval import load_dataset
from eval.sku_reranker.reranker import build_signature

try:
    from json_repair import repair_json
except ImportError:
    repair_json = None

try:
    from langchain_core.output_parsers import JsonOutputParser
except ImportError:
    JsonOutputParser = None
try:
    from langchain.chat_models import init_chat_model
except ImportError:
    init_chat_model = None
try:
    from langchain_deepseek import ChatDeepSeek
except ImportError:
    ChatDeepSeek = None
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


DATASET_PATH = ROOT_DIR / "data" / "eval" / "entity_linking.jsonl"
OUTPUT_PATH = ROOT_DIR / "data" / "eval" / "entity_linking_v2.jsonl"
SKU_REPANKER_DIR = ROOT_DIR / "logs" / "eval" / "sku_reranker"
LABEL_ORDER = ["SKU", "Trademark", "SPU", "Category3"]
DEFAULT_RATIO = {"SKU": 4, "Trademark": 3, "SPU": 3, "Category3": 3}
DEFAULT_TOTAL_TARGET = 1001
MAX_FULLTEXT_HIT = 20
MAX_HYBRID_HIT = 20
VALIDATION_TOP_K = 20


def scale_targets_from_total(total_target: int, ratio: dict[str, int]) -> dict[str, int]:
    total_weight = sum(ratio.get(label, 0) for label in LABEL_ORDER)
    if total_weight <= 0:
        return {label: 0 for label in LABEL_ORDER}
    raw_shares = {label: total_target * ratio.get(label, 0) / total_weight for label in LABEL_ORDER}
    targets = {label: int(math.floor(share)) for label, share in raw_shares.items()}
    remainder = total_target - sum(targets.values())
    ranked_labels = sorted(
        LABEL_ORDER,
        key=lambda label: (raw_shares[label] - targets[label], ratio.get(label, 0), -LABEL_ORDER.index(label)),
        reverse=True,
    )
    for label in ranked_labels[:remainder]:
        targets[label] += 1
    return targets


DEFAULT_TARGETS = scale_targets_from_total(DEFAULT_TOTAL_TARGET, DEFAULT_RATIO)


def _progress(iterable, *, total: int | None = None, desc: str = "", leave: bool = False):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=leave, dynamic_ncols=True, disable=not sys.stderr.isatty())


def _progress_bar(*, total: int, desc: str, leave: bool = False):
    if tqdm is None:
        return None
    return tqdm(total=total, desc=desc, leave=leave, dynamic_ncols=True, disable=not sys.stderr.isatty())


def resolve_local_embedding_model_name(model_name: str) -> str:
    candidate = Path(model_name)
    if candidate.exists():
        return str(candidate)
    if "/" not in model_name:
        return model_name
    owner, repo = model_name.split("/", 1)
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{owner}--{repo}" / "snapshots"
    if not cache_root.exists():
        return model_name
    snapshots = sorted([path for path in cache_root.iterdir() if path.is_dir()], reverse=True)
    for snapshot in snapshots:
        if (snapshot / "config.json").exists() and ((snapshot / "pytorch_model.bin").exists() or (snapshot / "model.safetensors").exists()):
            return str(snapshot)
    return model_name


@dataclass
class EntityRecord:
    label: str
    gold: str
    query: str
    source_entity_id: str | None = None
    source_entity_name: str | None = None
    variant_type: str = "seed"
    difficulty: str = "easy"
    source: str = "seed"
    validation_note: str = ""
    validation_exact_hit: bool = False
    validation_fulltext_hit: bool = False
    validation_hybrid_hit: bool = False
    validation_topk_hit: bool = False
    validation_status: str = "pending"
    metadata: dict[str, Any] | None = None


class SearchBackend:
    def __init__(self, *, validation_workers: int = 3):
        self.graph = Neo4jWriter()
        self._embedding_model = None
        self._vectors: dict[str, Any] = {}
        self.hybrid_error: str | None = None
        self.validation_workers = max(1, validation_workers)
        self._validation_executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.validation_workers)

    def close(self) -> None:
        self._validation_executor.shutdown(wait=True)
        self.graph.close()

    def exact_exists(self, label: str, query: str) -> bool:
        rows = self.graph.query(
            f"""
            MATCH (n:{label} {{name: $query}})
            RETURN n.name AS name
            """,
            query=query,
        )
        return any(row.get("name") == query for row in rows)

    def search(self, label: str, query: str, *, mode: str, top_k: int) -> list[str]:
        if mode == "exact_match":
            rows = self.graph.query(
                f"""
                MATCH (n:{label} {{name: $query}})
                RETURN n.name AS name
                """,
                query=query,
            )
            return [row["name"] for row in rows if row.get("name")]

        index_info = ENTITY_INDEX_CONFIG[label]
        if mode == "fulltext":
            rows = self.graph.query(
                """
                CALL db.index.fulltext.queryNodes($index_name, $query) YIELD node, score
                RETURN node.name AS name, score
                ORDER BY score DESC
                LIMIT $limit
                """,
                index_name=index_info["fulltext_index"],
                query=build_safe_fulltext_query(query),
                limit=top_k,
            )
            return [row["name"] for row in rows if row.get("name")]

        if mode == "hybrid":
            if self.hybrid_error is not None:
                return []
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_neo4j import Neo4jVector
            from neo4j_graphrag.types import SearchType

            try:
                if self._embedding_model is None:
                    model_name = resolve_local_embedding_model_name(
                        os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-base-zh-v1.5")
                    )
                    self._embedding_model = HuggingFaceEmbeddings(
                        model_name=model_name,
                        encode_kwargs={"normalize_embeddings": True},
                    )
                if label not in self._vectors:
                    self._vectors[label] = Neo4jVector.from_existing_index(
                        self._embedding_model,
                        url=NEO4J_CONFIG["uri"],
                        username=NEO4J_CONFIG["auth"][0],
                        password=NEO4J_CONFIG["auth"][1],
                        index_name=index_info["vector_index"],
                        keyword_index_name=index_info["fulltext_index"],
                        search_type=SearchType.HYBRID,
                    )
                results = self._vectors[label].similarity_search(query, k=top_k)
                return [item.page_content for item in results if getattr(item, "page_content", "")]
            except Exception as exc:
                self.hybrid_error = str(exc)
                return []

        raise ValueError(f"Unsupported search mode: {mode}")

    def search_bundle(self, label: str, query: str) -> dict[str, list[str]]:
        if self.validation_workers <= 1:
            return {
                "exact_match": self.search(label, query, mode="exact_match", top_k=1),
                "fulltext": self.search(label, query, mode="fulltext", top_k=MAX_FULLTEXT_HIT),
                "hybrid": self.search(label, query, mode="hybrid", top_k=MAX_HYBRID_HIT),
            }

        futures = {
            "exact_match": self._validation_executor.submit(self.search, label, query, mode="exact_match", top_k=1),
            "fulltext": self._validation_executor.submit(self.search, label, query, mode="fulltext", top_k=MAX_FULLTEXT_HIT),
            "hybrid": self._validation_executor.submit(self.search, label, query, mode="hybrid", top_k=MAX_HYBRID_HIT),
        }
        results: dict[str, list[str]] = {}
        for mode, future in futures.items():
            try:
                results[mode] = future.result()
            except Exception:
                results[mode] = []
        return results


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def kept_count(records: list[EntityRecord], label: str) -> int:
    return sum(1 for item in records if item.label == label and item.validation_status == "kept")


def normalize_query(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def load_alias_map() -> dict[str, dict[str, str]]:
    config_path = ROOT_DIR / "src" / "configuration" / "entity_normalization.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    aliases: dict[str, dict[str, str]] = {}
    for label, spec in payload.items():
        aliases[label] = dict(spec.get("aliases", {}))
    return aliases


def load_seed_queries(dataset_path: Path) -> dict[str, list[dict[str, Any]]]:
    seeds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in load_dataset(dataset_path):
        seeds[record["label"]].append(
            {
                "query": record["query"],
                "gold": record["gold"],
                "source": "entity_linking_v1",
            }
        )

    for filename in ("rerank_helped.jsonl", "rerank_hurt.jsonl", "rerank_still_wrong.jsonl"):
        for record in load_jsonl(SKU_REPANKER_DIR / filename):
            seeds["SKU"].append(
                {
                    "query": record["query"],
                    "gold": record["gold"],
                    "source": f"sku_reranker:{filename}",
                }
            )
    return seeds


def build_llm():
    model_name = os.getenv("DEEPSEEK_MODEL", DEEPSEEK_MODEL or "deepseek-chat").strip()
    model_provider = os.getenv("CHAT_MODEL_PROVIDER", "deepseek").strip()
    model_identifier = os.getenv("CHAT_MODEL", "").strip()
    model_base_url = (
        os.getenv("CHAT_MODEL_BASE_URL", "").strip()
        or os.getenv("OPENAI_BASE_URL", "").strip()
        or os.getenv("DEEPSEEK_BASE_URL", "").strip()
    )
    temperature = float(os.getenv("MODEL_TEMPERATURE", "0"))
    max_retries = int(os.getenv("MODEL_MAX_RETRIES", "2"))
    deepseek_thinking = os.getenv("DEEPSEEK_THINKING", "disabled").strip().lower()

    model_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "max_retries": max_retries,
    }
    if model_base_url:
        model_kwargs["base_url"] = model_base_url
    if deepseek_thinking in {"enabled", "disabled"}:
        model_kwargs["extra_body"] = {
            "thinking": {
                "type": deepseek_thinking,
            }
        }

    if init_chat_model is not None:
        if model_identifier:
            return init_chat_model(model=model_identifier, **model_kwargs)
        return init_chat_model(model=model_name, model_provider=model_provider, **model_kwargs)

    if ChatDeepSeek is not None:
        chat_kwargs = {
            "model": model_identifier or model_name,
            "api_key": DEEPSEEK_API_KEY,
            "temperature": temperature,
            "max_retries": max_retries,
        }
        if model_base_url:
            chat_kwargs["base_url"] = model_base_url
        if "extra_body" in model_kwargs:
            chat_kwargs["extra_body"] = model_kwargs["extra_body"]
        return ChatDeepSeek(**chat_kwargs)

    raise ImportError("No compatible chat model initializer is available in the current environment.")


def fetch_entity_pool(graph: Neo4jWriter, label: str, limit: int | None = None) -> list[dict[str, Any]]:
    query_limit = limit or 1000
    rows = graph.query(
        f"""
        MATCH (n:{label})
        RETURN n.id AS entity_id, n.name AS name
        ORDER BY toString(n.name)
        LIMIT $limit
        """,
        limit=query_limit,
    )
    return [row for row in rows if row.get("name")]


def load_entity_pools(*, limit_by_label: dict[str, int] | None = None) -> dict[str, list[dict[str, Any]]]:
    graph = Neo4jWriter()
    try:
        pools = {}
        for label in LABEL_ORDER:
            pools[label] = fetch_entity_pool(graph, label, limit=(limit_by_label or {}).get(label))
        return pools
    finally:
        graph.close()


def _split_tokens(text: str) -> list[str]:
    text = normalize_query(text)
    if not text:
        return []
    return [item for item in re.split(r"[\s_]+", text) if item]


def _compact(text: str) -> str:
    return re.sub(r"[\s\-_()（）/，,。:：;；]+", "", text)


def _insert_space_between_alnum(text: str) -> str:
    compact = _compact(text)
    compact = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", compact)
    compact = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", compact)
    compact = re.sub(r"\s+", " ", compact).strip()
    return compact


def _variant_difficulty(label: str, query: str, seed_query: str, gold: str) -> str:
    normalized_query = normalize_query(query)
    normalized_seed = normalize_query(seed_query)
    normalized_gold = normalize_query(gold)
    if normalized_query in {normalized_seed, normalized_gold}:
        return "easy"
    if label == "SKU":
        compact_seed = normalize_query(_compact(seed_query))
        compact_gold = normalize_query(_compact(gold))
        if normalized_query in {compact_seed, compact_gold}:
            return "hard"
        if len(_split_tokens(query)) <= 3:
            return "hard"
        if re.search(r"\d", normalized_query) and re.search(r"[a-z]", normalized_query):
            return "medium"
        return "hard"
    if label in {"Trademark", "SPU"}:
        return "medium"
    if label == "Category3":
        return "medium"
    return "medium"


def build_local_variants(label: str, gold: str, seed_query: str, aliases: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    normalized_gold = normalize_query(gold)
    normalized_seed = normalize_query(seed_query)
    seed_tokens = _split_tokens(seed_query)
    gold_tokens = _split_tokens(gold)
    alias_map = aliases.get(label, {})

    candidates = [
        seed_query,
        normalized_seed,
        _compact(seed_query),
        _compact(gold),
        _insert_space_between_alnum(seed_query),
        _insert_space_between_alnum(gold),
        " ".join(seed_tokens),
        " ".join(reversed(seed_tokens)),
        " ".join(gold_tokens[: max(1, len(gold_tokens) // 2)]),
    ]

    if label == "Trademark":
        for token in list(seed_tokens)[:2]:
            candidates.extend([token, token.upper(), token.lower()])
            if re.fullmatch(r"[a-z]{6,}", token):
                split_at = max(3, len(token) // 2)
                candidates.append(f"{token[:split_at]} {token[split_at:]}")
        candidates.extend(alias_map.keys())
    elif label == "SPU":
        candidates.extend(
            [
                " ".join(gold_tokens[:2]),
                " ".join(gold_tokens[-2:]),
                _compact(" ".join(gold_tokens[:3])),
                " ".join(token for token in gold_tokens if token not in {"apple", "华为", "redmi", "xiaomi"}),
            ]
        )
    elif label == "Category3":
        candidates.extend([gold, normalized_gold, _compact(gold), gold.replace(" ", ""), gold.replace("配件", " 配件")])
    elif label == "SKU":
        signature = build_signature(gold)
        query_signature = build_signature(seed_query)
        compact_gold = _compact(gold)
        candidates.extend(
            [
                compact_gold,
                " ".join(seed_tokens[:3]),
                " ".join(seed_tokens[-3:]),
                _compact(seed_query),
                " ".join(seed_tokens[:2] + seed_tokens[-2:]),
                " ".join(gold_tokens[:4]),
                " ".join(gold_tokens[-4:]),
            ]
        )
        if signature.model_tokens:
            candidates.extend(sorted(signature.model_tokens))
        if signature.capacity_tokens:
            candidates.extend(sorted(signature.capacity_tokens))
        if query_signature.capacity_tokens:
            candidates.extend(sorted(query_signature.capacity_tokens))
        if signature.model_tokens and signature.capacity_tokens:
            for model_token in sorted(signature.model_tokens):
                for capacity_token in sorted(signature.capacity_tokens):
                    candidates.append(f"{model_token} {capacity_token}")
                    candidates.append(f"{capacity_token} {model_token}")
        if seed_tokens:
            candidates.append(" ".join(token for token in seed_tokens if token not in {"官方", "旗舰", "手机", "电视", "平板", "支持"}))
    else:
        candidates.extend([gold, normalized_gold])

    deduped = []
    seen = set()
    for item in candidates:
        query = str(item or "").strip()
        if not query:
            continue
        key = normalize_query(query)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(
            {
                "query": query,
                "variant_type": "heuristic",
                "difficulty": _variant_difficulty(label, query, seed_query, gold),
                "source": "heuristic",
            }
        )
    return deduped


def _parse_llm_payload(raw_content: str) -> dict[str, Any] | None:
    if JsonOutputParser is None:
        try:
            return json.loads(raw_content)
        except json.JSONDecodeError:
            return None

    parser = JsonOutputParser()
    try:
        payload = parser.invoke(raw_content)
    except Exception:
        repaired = repair_json(raw_content, ensure_ascii=False) if repair_json is not None else raw_content
        try:
            payload = parser.invoke(repaired)
        except Exception:
            return None
    return payload if isinstance(payload, dict) else None


def generate_llm_variants(
    *,
    llm,
    label: str,
    gold: str,
    seed_query: str,
    entity_id: str | None,
    batch_size: int,
) -> list[dict[str, Any]]:
    if llm is None:
        return []
    prompt = f"""
你在构造电商实体链接评测集，目标实体标签是 {label}。
已确认金标实体：{gold}
实体来源 id：{entity_id or "unknown"}
种子查询：{seed_query}

请生成 {batch_size} 条新的 query 变体，只输出严格 JSON：
{{
  "items": [
    {{
      "query": "query 文本",
      "difficulty": "easy|medium|hard",
      "variant_type": "alias|abbrev|reorder|hard_negative|compression",
      "validation_note": "为什么这个 query 仍然指向该 gold"
    }}
  ]
}}

要求：
1. query 必须仍然指向同一个 gold。
2. 不要输出与种子完全相同的 query。
3. SKU 优先生成容量、颜色、型号、品牌顺序变化和压缩搜索式。
4. Trademark / SPU / Category3 优先生成大小写、空格、中英混写、别名、顺序变化。
5. 不要输出解释文字。
"""
    response = llm.invoke(prompt)
    raw_content = getattr(response, "content", str(response))
    payload = _parse_llm_payload(raw_content)
    if not payload:
        return []
    items = payload.get("items", [])
    if not isinstance(items, list):
        return []
    normalized = []
    for item in items:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query", "")).strip()
        if not query:
            continue
        normalized.append(
            {
                "query": query,
                "variant_type": str(item.get("variant_type", "llm")).strip() or "llm",
                "difficulty": str(item.get("difficulty", "medium")).strip() or "medium",
                "source": "llm",
                "validation_note": str(item.get("validation_note", "")).strip(),
            }
        )
    return normalized


def prefetch_llm_variants(
    *,
    llm,
    jobs: list[dict[str, Any]],
    max_workers: int,
    batch_size: int,
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    if llm is None or not jobs:
        return {}
    if max_workers <= 1:
        return {
            (job["label"], job["gold"], job["seed_query"]): generate_llm_variants(
                llm=llm,
                label=job["label"],
                gold=job["gold"],
                seed_query=job["seed_query"],
                entity_id=job["entity_id"],
                batch_size=batch_size,
            )
            for job in jobs
        }

    results: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    llm_progress = _progress_bar(total=len(jobs), desc="LLM batches", leave=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {}
        for job in jobs:
            key = (job["label"], job["gold"], job["seed_query"])
            future = executor.submit(
                generate_llm_variants,
                llm=llm,
                label=job["label"],
                gold=job["gold"],
                seed_query=job["seed_query"],
                entity_id=job["entity_id"],
                batch_size=batch_size,
            )
            future_to_key[future] = key
        for future in concurrent.futures.as_completed(future_to_key):
            key = future_to_key[future]
            try:
                results[key] = future.result()
            except Exception:
                results[key] = []
            if llm_progress is not None:
                llm_progress.update(1)
    if llm_progress is not None:
        llm_progress.close()
    return results


def build_validation_index(pools: dict[str, list[dict[str, Any]]]) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for label, rows in pools.items():
        index[label] = {normalize_query(row["name"]) for row in rows if row.get("name")}
    return index


def validate_record(
    record: EntityRecord,
    *,
    searcher: SearchBackend,
    validation_index: dict[str, set[str]],
    top_k: int = VALIDATION_TOP_K,
) -> EntityRecord:
    exact_exists = normalize_query(record.gold) in validation_index.get(record.label, set())
    bundle = searcher.search_bundle(record.label, record.query)
    exact_hit = bundle["exact_match"] == [record.gold]
    fulltext_candidates = bundle["fulltext"]
    hybrid_candidates = bundle["hybrid"]
    record.validation_exact_hit = exact_hit
    record.validation_fulltext_hit = record.gold in fulltext_candidates[:top_k]
    record.validation_hybrid_hit = record.gold in hybrid_candidates[:top_k]
    record.validation_topk_hit = record.validation_fulltext_hit or record.validation_hybrid_hit
    if not exact_exists:
        record.validation_status = "invalid_gold"
        record.validation_note = record.validation_note or "gold_missing_in_graph"
    elif record.validation_topk_hit:
        record.validation_status = "kept"
    else:
        record.validation_status = "rejected"
        record.validation_note = record.validation_note or "gold_not_retrievable_in_topk"
    if searcher.hybrid_error and not record.validation_hybrid_hit:
        note = f"hybrid_unavailable:{searcher.hybrid_error}"
        if record.validation_note:
            if note not in record.validation_note:
                record.validation_note = f"{record.validation_note}; {note}"
        else:
            record.validation_note = note
    return record


def _select_seed_query(label: str, gold: str, seeds: dict[str, list[dict[str, Any]]]) -> tuple[str, str]:
    label_seeds = seeds.get(label, [])
    if not label_seeds:
        return gold, "entity_linking_v1"
    for seed in label_seeds:
        if seed["gold"] == gold:
            return seed["query"], seed["source"]
    seed = label_seeds[0]
    return seed["query"], seed["source"]


def _build_record(label: str, gold: str, query: str, *, source_entity_id: str | None, source_entity_name: str | None, variant_type: str, difficulty: str, source: str, validation_note: str = "", metadata: dict[str, Any] | None = None) -> EntityRecord:
    return EntityRecord(
        label=label,
        gold=gold,
        query=query,
        source_entity_id=source_entity_id,
        source_entity_name=source_entity_name,
        variant_type=variant_type,
        difficulty=difficulty,
        source=source,
        validation_note=validation_note,
        metadata=metadata or {},
    )


def generate_dataset(
    *,
    dataset_path: Path,
    target_by_label: dict[str, int],
    llm_enabled: bool,
    max_llm_items_per_seed: int,
    llm_workers: int = 4,
    validation_workers: int = 3,
    limit_by_label: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    seeds = load_seed_queries(dataset_path)
    aliases = load_alias_map()
    pools = load_entity_pools(limit_by_label=limit_by_label)
    validation_index = build_validation_index(pools)
    searcher = SearchBackend(validation_workers=validation_workers)
    llm = None
    if llm_enabled and DEEPSEEK_API_KEY:
        try:
            llm = build_llm()
        except Exception:
            llm = None

    try:
        records: list[EntityRecord] = []
        stats = defaultdict(int)
        source_lookup = {label: {row["name"]: row.get("entity_id") for row in rows} for label, rows in pools.items()}
        old_records = load_dataset(dataset_path)
        old_pairs = {(normalize_query(item["query"]), item["label"]) for item in old_records}
        seen_pairs = set(old_pairs)

        for item in _progress(old_records, total=len(old_records), desc="Seed records", leave=False):
            entity_id = source_lookup.get(item["label"], {}).get(item["gold"])
            preserved = _build_record(
                label=item["label"],
                gold=item["gold"],
                query=item["query"],
                source_entity_id=str(entity_id) if entity_id is not None else None,
                source_entity_name=item["gold"],
                variant_type="seed_v1",
                difficulty="easy",
                source="entity_linking_v1",
                metadata={"preserved_from_v1": True},
            )
            records.append(validate_record(preserved, searcher=searcher, validation_index=validation_index))
            stats[(item["label"], "seed_v1")] += 1

        priority_gold = {label: [seed["gold"] for seed in seeds.get(label, [])] for label in LABEL_ORDER}

        llm_jobs: list[dict[str, Any]] = []
        label_progress = _progress_bar(total=len(LABEL_ORDER), desc="Labels", leave=False)
        for label in LABEL_ORDER:
            target = target_by_label.get(label, DEFAULT_TARGETS[label])
            entity_rows = sorted(
                pools.get(label, []),
                key=lambda row: (
                    0 if row.get("name") in priority_gold.get(label, []) else 1,
                    str(row.get("name") or ""),
                ),
            )
            row_progress = _progress_bar(total=len(entity_rows), desc=f"{label} pool", leave=False)
            for row in entity_rows:
                if kept_count(records, label) >= target:
                    break
                gold = str(row.get("name"))
                entity_id = str(row.get("entity_id")) if row.get("entity_id") is not None else None
                seed_query, seed_source = _select_seed_query(label, gold, seeds)
                base_query = seed_query if seed_query else gold
                base = _build_record(
                    label=label,
                    gold=gold,
                    query=base_query,
                    source_entity_id=entity_id,
                    source_entity_name=gold,
                    variant_type="seed",
                    difficulty="easy",
                    source=seed_source,
                    metadata={"entity_pool_rank": str(row.get("name"))},
                )
                if (normalize_query(base.query), base.label) not in seen_pairs:
                    seen_pairs.add((normalize_query(base.query), base.label))
                    records.append(validate_record(base, searcher=searcher, validation_index=validation_index))
                    stats[(label, "seed")] += 1

                if llm is None:
                    local_variants = build_local_variants(label, gold, base_query, aliases)
                    for variant in local_variants:
                        if kept_count(records, label) >= target:
                            break
                        pair = (normalize_query(variant["query"]), label)
                        if pair in seen_pairs:
                            continue
                        seen_pairs.add(pair)
                        candidate = _build_record(
                            label=label,
                            gold=gold,
                            query=variant["query"],
                            source_entity_id=entity_id,
                            source_entity_name=gold,
                            variant_type=variant["variant_type"],
                            difficulty=variant["difficulty"],
                            source=variant["source"],
                        )
                        records.append(validate_record(candidate, searcher=searcher, validation_index=validation_index))
                        stats[(label, candidate.variant_type)] += 1

                if llm is not None and kept_count(records, label) < target:
                    llm_jobs.append(
                        {
                            "label": label,
                            "gold": gold,
                            "seed_query": base_query,
                            "entity_id": entity_id,
                        }
                    )
                if row_progress is not None:
                    row_progress.update(1)
            if row_progress is not None:
                row_progress.close()
            if label_progress is not None:
                label_progress.update(1)
        if label_progress is not None:
            label_progress.close()

        llm_variants_by_job = prefetch_llm_variants(
            llm=llm,
            jobs=llm_jobs,
            max_workers=llm_workers,
            batch_size=max_llm_items_per_seed,
        )
        llm_apply_progress = _progress_bar(total=len(llm_jobs), desc="Apply LLM", leave=False)
        for job in llm_jobs:
            label = job["label"]
            target = target_by_label.get(label, DEFAULT_TARGETS[label])
            if kept_count(records, label) >= target:
                if llm_apply_progress is not None:
                    llm_apply_progress.update(1)
                continue
            key = (job["label"], job["gold"], job["seed_query"])
            entity_id = job["entity_id"]
            gold = job["gold"]
            for variant in llm_variants_by_job.get(key, []):
                if kept_count(records, label) >= target:
                    break
                pair = (normalize_query(variant["query"]), label)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                candidate = _build_record(
                    label=label,
                    gold=gold,
                    query=variant["query"],
                    source_entity_id=entity_id,
                    source_entity_name=gold,
                    variant_type=variant["variant_type"],
                    difficulty=variant["difficulty"],
                    source=variant["source"],
                    validation_note=variant.get("validation_note", ""),
                )
                records.append(validate_record(candidate, searcher=searcher, validation_index=validation_index))
                stats[(label, candidate.variant_type)] += 1
            if llm_apply_progress is not None:
                llm_apply_progress.update(1)
        if llm_apply_progress is not None:
            llm_apply_progress.close()

        exported_records = list(records)
        review_records = [item for item in records if item.validation_status != "kept"]

        summary = {
            "targets": target_by_label,
            "counts": {label: sum(1 for item in exported_records if item.label == label) for label in LABEL_ORDER},
            "review_counts": {label: sum(1 for item in review_records if item.label == label) for label in LABEL_ORDER},
            "stats": {f"{label}:{kind}": count for (label, kind), count in stats.items()},
            "validation_status_counts": {
                status: sum(1 for item in records if item.validation_status == status)
                for status in sorted({item.validation_status for item in records})
            },
            "source_counts": {
                label: {
                    "seed": sum(1 for item in exported_records if item.label == label and item.source.startswith("entity_linking_v1")),
                    "sku_reranker": sum(1 for item in exported_records if item.label == label and item.source.startswith("sku_reranker")),
                    "heuristic": sum(1 for item in exported_records if item.label == label and item.source == "heuristic"),
                    "llm": sum(1 for item in exported_records if item.label == label and item.source == "llm"),
                }
                for label in LABEL_ORDER
            },
            "review_reason_samples": {
                label: [item.validation_note for item in review_records if item.label == label][:5]
                for label in LABEL_ORDER
            },
        }
        return [item.__dict__ for item in exported_records], [item.__dict__ for item in review_records], summary
    finally:
        searcher.close()


def parse_target_args(values: list[str] | None, *, base_targets: dict[str, int] | None = None) -> dict[str, int]:
    targets = dict(base_targets or DEFAULT_TARGETS)
    if not values:
        return targets
    for value in values:
        if "=" not in value:
            continue
        label, raw_count = value.split("=", 1)
        label = label.strip()
        if label not in targets:
            continue
        try:
            targets[label] = int(raw_count)
        except ValueError:
            continue
    return targets


def parse_limit_args(values: list[str] | None) -> dict[str, int]:
    if not values:
        return {}
    limits: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            continue
        label, raw_count = value.split("=", 1)
        label = label.strip()
        try:
            limits[label] = int(raw_count)
        except ValueError:
            continue
    return limits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an expanded entity linking dataset.")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--review-output", default="")
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--total-target", type=int, default=DEFAULT_TOTAL_TARGET)
    parser.add_argument(
        "--target",
        action="append",
        help="Override target count, e.g. --target SKU=200",
    )
    parser.add_argument(
        "--limit",
        action="append",
        help="Limit graph pool per label, e.g. --limit SKU=100",
    )
    parser.add_argument("--no-llm", action="store_true", help="Disable DeepSeek generation.")
    parser.add_argument("--llm-batch-size", type=int, default=3)
    parser.add_argument("--llm-workers", type=int, default=4)
    parser.add_argument("--validation-workers", type=int, default=3)
    parser.add_argument("--summary", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_by_label = scale_targets_from_total(max(1, args.total_target), DEFAULT_RATIO)
    target_by_label = parse_target_args(args.target, base_targets=target_by_label)
    limit_by_label = parse_limit_args(args.limit)
    records, review_records, summary = generate_dataset(
        dataset_path=Path(args.dataset),
        target_by_label=target_by_label,
        llm_enabled=not args.no_llm,
        max_llm_items_per_seed=max(1, args.llm_batch_size),
        llm_workers=max(1, args.llm_workers),
        validation_workers=max(1, args.validation_workers),
        limit_by_label=limit_by_label,
    )
    write_jsonl(Path(args.output), records)
    review_output = Path(args.review_output) if args.review_output else Path(args.output).with_suffix(".review.jsonl")
    write_jsonl(review_output, review_records)
    summary_path = Path(args.summary) if args.summary else Path(args.output).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
