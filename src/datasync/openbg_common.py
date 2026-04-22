import hashlib
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
OPENBG_JSONL_FILE = ROOT_DIR / "data" / "OpenBG-Align_item_info_train_dev_sample500_per_industry.jsonl"
SKU_SALE_ATTR_WHITELIST = {
    "颜色分类",
    "颜色",
    "尺码",
    "尺寸",
    "套餐类型",
    "规格",
    "口味",
    "食品口味",
    "净含量",
    "化妆品净含量",
    "香味",
    "版本类型",
    "存储容量",
    "机身颜色",
    "内存容量",
    "机身内存ROM",
    "网络类型",
    "长度",
}
PVS_SPLITTER = "#;#"
PVS_KV_SPLITTER = "#:#"
TEXT_NORMALIZER = re.compile(r"[\s/／、,，:：;；【】\[\]\(\)（）\-+]+")


@dataclass
class Category3Candidate:
    id: object
    name: str
    parent_id: object
    parent_name: str
    source: str


@dataclass
class Category2Candidate:
    id: object
    name: str


def load_openbg_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_pvs(raw_value: str | None) -> list[tuple[str, str]]:
    pairs = []
    for part in (raw_value or "").split(PVS_SPLITTER):
        if PVS_KV_SPLITTER not in part:
            continue
        key, value = part.split(PVS_KV_SPLITTER, 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            pairs.append((key, value))
    return pairs


def split_category_path(raw_path: str | None) -> list[str]:
    return [segment.strip() for segment in (raw_path or "").split("->") if segment.strip()]


def normalize_text(text: str | None) -> str:
    value = (text or "").strip().lower()
    value = TEXT_NORMALIZER.sub("", value)
    return value


def similarity(left: str | None, right: str | None) -> float:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    if left_norm in right_norm or right_norm in left_norm:
        return 0.96
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def choose_best_category3(
    cate_name: str,
    cate_name_path: str,
    candidates: list[Category3Candidate],
) -> tuple[Category3Candidate | None, float]:
    path_segments = split_category_path(cate_name_path)
    parent_hint = path_segments[-2] if len(path_segments) >= 2 else ""

    best_candidate = None
    best_score = 0.0
    for candidate in candidates:
        name_score = similarity(cate_name, candidate.name)
        parent_score = similarity(parent_hint, candidate.parent_name)
        score = name_score * 0.85 + parent_score * 0.15
        if score > best_score:
            best_candidate = candidate
            best_score = score

    return best_candidate, best_score


def choose_best_category2(
    cate_name: str,
    cate_name_path: str,
    industry_name: str,
    candidates: list[Category2Candidate],
) -> tuple[Category2Candidate | None, float]:
    path_segments = split_category_path(cate_name_path)
    parent_hint = path_segments[-2] if len(path_segments) >= 2 else ""
    best_candidate = None
    best_score = 0.0
    for candidate in candidates:
        parent_score = similarity(parent_hint, candidate.name)
        cate_score = similarity(cate_name, candidate.name)
        industry_score = similarity(industry_name, candidate.name)
        score = max(parent_score, cate_score * 0.7, industry_score * 0.55)
        if score > best_score:
            best_candidate = candidate
            best_score = score
    return best_candidate, best_score


def normalize_brand_name(raw_brand: str | None) -> str:
    value = (raw_brand or "").strip()
    if not value:
        return ""

    parts = [part.strip() for part in re.split(r"[/／]", value) if part.strip()]
    if not parts:
        return value

    chinese_parts = [part for part in parts if re.search(r"[\u4e00-\u9fff]", part)]
    if chinese_parts:
        value = chinese_parts[-1]
    else:
        value = parts[0]

    normalized = normalize_text(value)
    if normalized in {"其他", "other", "na", "n/a", "无", "未知"}:
        return ""
    return value


def extract_brand(pairs: list[tuple[str, str]]) -> str:
    for key, value in pairs:
        if "品牌" in key:
            return normalize_brand_name(value)
    return ""


def stable_id(prefix: str, raw_value: str) -> str:
    digest = hashlib.md5(raw_value.encode("utf-8")).hexdigest()
    return f"{prefix}::{digest}"
