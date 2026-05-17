from __future__ import annotations

import re
from dataclasses import dataclass


WHITESPACE_PATTERN = re.compile(r"\s+")
SEPARATOR_PATTERN = re.compile(r"[\(\)\[\]\{\}/_,，。:：;；\-]+")
TOKEN_PATTERN = re.compile(r"[a-z0-9+]+|[\u4e00-\u9fff]+")
MODEL_TOKEN_PATTERN = re.compile(r"^(?=.*[a-z])(?=.*\d)[a-z0-9+]+$")
CAPACITY_TOKEN_PATTERN = re.compile(r"^\d+(?:gb|g|kg|ml|l)$|^\d+\+\d+gb$|^\d+gb\+\d+gb$")
ALNUM_SEGMENT_PATTERN = re.compile(r"[a-z]+|\d+[a-z]*")
STORAGE_GB_PATTERN = re.compile(r"^(\d+)gb$")
WEIGHT_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)(kg|g)$")
VOLUME_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)(ml|l)$")
COUNT_PATTERN = re.compile(r"^(\d+)(片|件|包|袋|支|盒)$")
COLOR_KEYWORDS = (
    "黑",
    "白",
    "红",
    "蓝",
    "灰",
    "金",
    "银",
    "紫",
    "粉",
    "绿",
    "黄",
    "橙",
    "棕",
    "冰雾白",
    "明月灰",
)


@dataclass
class RerankFeatures:
    model_match: bool
    capacity_match: bool
    color_match: bool
    brand_match: bool
    subject_match: bool
    conflict_tokens: int


@dataclass
class RerankWeights:
    model: float = 10.0
    capacity: float = 8.0
    color: float = 3.0
    brand: float = 2.0
    subject: float = 6.0
    conflict_penalty: float = -5.0


@dataclass
class SKUSignature:
    tokens: list[str]
    model_tokens: set[str]
    capacity_tokens: set[str]
    color_tokens: set[str]
    brand_tokens: set[str]
    subject_tokens: set[str]


@dataclass
class CapacitySignals:
    storage_gb: int | None = None
    weight_kg: float | None = None
    volume_ml: float | None = None
    count_packs: tuple[int, ...] | None = None


@dataclass
class ScoredCandidate:
    candidate: str
    score: float
    features: RerankFeatures
    original_index: int


def _normalize_text(text: str) -> str:
    normalized = str(text or "").lower()
    normalized = normalized.replace("（", "(").replace("）", ")")
    normalized = normalized.replace("×", "x").replace("＊", "*")
    normalized = WHITESPACE_PATTERN.sub(" ", normalized).strip()
    normalized = SEPARATOR_PATTERN.sub(" ", normalized)
    return normalized


def _tokenize(text: str) -> list[str]:
    tokens = []
    for token in TOKEN_PATTERN.findall(_normalize_text(text)):
        if re.fullmatch(r"[a-z]{3,}\d+[a-z]*", token):
            segments = [segment for segment in ALNUM_SEGMENT_PATTERN.findall(token) if segment]
            if len(segments) > 1:
                tokens.extend(segments)
                continue
        prefix_split = re.match(r"^([a-z]{3,})(.+)$", token)
        if prefix_split is not None and re.search(r"\d", prefix_split.group(2)):
            tokens.append(prefix_split.group(1))
            tokens.append(prefix_split.group(2))
            continue
        tokens.append(token)
    return tokens


def _extract_color_tokens(tokens: list[str]) -> set[str]:
    hits = set()
    for token in tokens:
        for color in COLOR_KEYWORDS:
            if color in token:
                hits.add(color)
    return hits


def _extract_capacity_tokens(tokens: list[str]) -> set[str]:
    hits = set()
    for token in tokens:
        compact = token.replace(" ", "")
        if CAPACITY_TOKEN_PATTERN.match(compact):
            hits.add(compact)
    return hits


def _extract_model_tokens(tokens: list[str], capacity_tokens: set[str]) -> set[str]:
    hits = set()
    for token in tokens:
        if token in capacity_tokens:
            continue
        if MODEL_TOKEN_PATTERN.match(token):
            hits.add(token)
    return hits


def _extract_brand_tokens(
    tokens: list[str],
    *,
    model_tokens: set[str],
    capacity_tokens: set[str],
    color_tokens: set[str],
) -> set[str]:
    brand_tokens = []
    for token in tokens:
        if token in model_tokens or token in capacity_tokens:
            continue
        if token in color_tokens:
            continue
        if token.isdigit():
            continue
        brand_tokens.append(token)
    return set(brand_tokens[:2])


def _extract_subject_tokens(
    tokens: list[str],
    *,
    model_tokens: set[str],
    capacity_tokens: set[str],
    color_tokens: set[str],
    brand_tokens: set[str],
) -> set[str]:
    subject_tokens = set()
    for token in tokens:
        if token in model_tokens or token in capacity_tokens:
            continue
        if token in color_tokens:
            continue
        if token.isdigit():
            continue
        if len(token) < 2 and not re.search(r"[a-z]", token):
            continue
        subject_tokens.add(token)
    return subject_tokens


def _parse_capacity_signals(tokens: list[str]) -> CapacitySignals:
    storage_gb = None
    weight_kg = None
    volume_ml = None
    count_values: list[int] = []
    for token in tokens:
        compact = token.replace(" ", "")
        if STORAGE_GB_PATTERN.match(compact):
            storage_gb = int(STORAGE_GB_PATTERN.match(compact).group(1))
            continue
        if compact.count("+") == 1 and compact.endswith("gb"):
            parts = compact[:-2].split("+")
            if len(parts) == 2 and all(part.isdigit() for part in parts):
                count_values.extend(int(part) for part in parts)
                continue
        weight_match = WEIGHT_PATTERN.match(compact)
        if weight_match:
            value = float(weight_match.group(1))
            unit = weight_match.group(2)
            weight_kg = value if unit == "kg" else value / 1000.0
            continue
        volume_match = VOLUME_PATTERN.match(compact)
        if volume_match:
            value = float(volume_match.group(1))
            unit = volume_match.group(2)
            volume_ml = value * 1000.0 if unit == "l" else value
            continue
        count_match = COUNT_PATTERN.match(compact)
        if count_match:
            count_values.append(int(count_match.group(1)))
    return CapacitySignals(
        storage_gb=storage_gb,
        weight_kg=weight_kg,
        volume_ml=volume_ml,
        count_packs=tuple(count_values) if count_values else None,
    )


def build_signature(text: str) -> SKUSignature:
    tokens = _tokenize(text)
    capacity_tokens = _extract_capacity_tokens(tokens)
    model_tokens = _extract_model_tokens(tokens, capacity_tokens)
    color_tokens = _extract_color_tokens(tokens)
    brand_tokens = _extract_brand_tokens(
        tokens,
        model_tokens=model_tokens,
        capacity_tokens=capacity_tokens,
        color_tokens=color_tokens,
    )
    subject_tokens = _extract_subject_tokens(
        tokens,
        model_tokens=model_tokens,
        capacity_tokens=capacity_tokens,
        color_tokens=color_tokens,
        brand_tokens=brand_tokens,
    )
    return SKUSignature(
        tokens=tokens,
        model_tokens=model_tokens,
        capacity_tokens=capacity_tokens,
        color_tokens=color_tokens,
        brand_tokens=brand_tokens,
        subject_tokens=subject_tokens,
    )


def _has_match(left: set[str], right: set[str]) -> bool:
    return bool(left and right and left.intersection(right))


def _meaningful_brand_overlap(left: set[str], right: set[str]) -> bool:
    if not left or not right:
        return False
    for token in left.intersection(right):
        if re.search(r"[\u4e00-\u9fff]", token) and len(token) >= 2:
            return True
        if re.search(r"[a-z]", token) and len(token) >= 4:
            return True
    return False


def _count_conflicts(left: set[str], right: set[str]) -> int:
    if not left or not right:
        return 0
    if left.intersection(right):
        return 0
    return 1


def _count_missing_context(query_signature: SKUSignature, candidate_signature: SKUSignature) -> int:
    if query_signature.model_tokens and not candidate_signature.model_tokens and not candidate_signature.brand_tokens:
        return 1
    if query_signature.brand_tokens and not candidate_signature.brand_tokens and not candidate_signature.model_tokens:
        return 1
    return 0


def _capacity_dimension_match(query_signals: CapacitySignals, candidate_signals: CapacitySignals) -> bool:
    if query_signals.storage_gb is not None and candidate_signals.storage_gb is not None:
        return query_signals.storage_gb == candidate_signals.storage_gb
    if query_signals.weight_kg is not None and candidate_signals.weight_kg is not None:
        return query_signals.weight_kg == candidate_signals.weight_kg
    if query_signals.volume_ml is not None and candidate_signals.volume_ml is not None:
        return query_signals.volume_ml == candidate_signals.volume_ml
    if query_signals.count_packs is not None and candidate_signals.count_packs is not None:
        return query_signals.count_packs == candidate_signals.count_packs
    return False


def _capacity_dimension_conflict(query_signals: CapacitySignals, candidate_signals: CapacitySignals) -> int:
    if query_signals.storage_gb is not None and candidate_signals.storage_gb is not None:
        return int(query_signals.storage_gb != candidate_signals.storage_gb)
    if query_signals.weight_kg is not None and candidate_signals.weight_kg is not None:
        return int(query_signals.weight_kg != candidate_signals.weight_kg)
    if query_signals.volume_ml is not None and candidate_signals.volume_ml is not None:
        return int(query_signals.volume_ml != candidate_signals.volume_ml)
    if query_signals.count_packs is not None and candidate_signals.count_packs is not None:
        return int(query_signals.count_packs != candidate_signals.count_packs)
    return 0


def _subject_gate_conflict(query_signature: SKUSignature, candidate_signature: SKUSignature) -> int:
    if not query_signature.subject_tokens:
        return 0
    if query_signature.subject_tokens.intersection(candidate_signature.subject_tokens):
        return 0
    return 1


class RuleBasedSKUReranker:
    def __init__(self, *, weights: RerankWeights | None = None):
        self.weights = weights or RerankWeights()

    def extract_features(self, query: str, candidate: str) -> RerankFeatures:
        query_signature = build_signature(query)
        candidate_signature = build_signature(candidate)
        query_capacity = _parse_capacity_signals(query_signature.capacity_tokens)
        candidate_capacity = _parse_capacity_signals(candidate_signature.capacity_tokens)
        model_match = _has_match(query_signature.model_tokens, candidate_signature.model_tokens)
        raw_capacity_match = _capacity_dimension_match(query_capacity, candidate_capacity)
        color_match = _has_match(query_signature.color_tokens, candidate_signature.color_tokens)
        brand_match = _meaningful_brand_overlap(query_signature.brand_tokens, candidate_signature.brand_tokens)
        subject_match = _has_match(query_signature.subject_tokens, candidate_signature.subject_tokens)
        context_match = model_match or brand_match
        capacity_match = raw_capacity_match and context_match
        conflict_tokens = 0
        conflict_tokens += _count_conflicts(query_signature.model_tokens, candidate_signature.model_tokens)
        conflict_tokens += _capacity_dimension_conflict(query_capacity, candidate_capacity)
        conflict_tokens += _count_conflicts(query_signature.color_tokens, candidate_signature.color_tokens)
        conflict_tokens += _count_conflicts(query_signature.brand_tokens, candidate_signature.brand_tokens)
        conflict_tokens += _count_missing_context(query_signature, candidate_signature)
        conflict_tokens += _subject_gate_conflict(query_signature, candidate_signature)
        return RerankFeatures(
            model_match=model_match,
            capacity_match=capacity_match,
            color_match=color_match,
            brand_match=brand_match,
            subject_match=subject_match,
            conflict_tokens=conflict_tokens,
        )

    def score_candidate(self, query: str, candidate: str, *, original_index: int = 0) -> ScoredCandidate:
        features = self.extract_features(query, candidate)
        score = 0.0
        score += self.weights.model if features.model_match else 0.0
        score += self.weights.capacity if features.capacity_match else 0.0
        score += self.weights.color if features.color_match else 0.0
        score += self.weights.brand if features.brand_match else 0.0
        score += self.weights.subject if features.subject_match else 0.0
        score += self.weights.conflict_penalty * float(features.conflict_tokens)
        return ScoredCandidate(
            candidate=candidate,
            score=score,
            features=features,
            original_index=original_index,
        )

    def score_candidates(self, query: str, candidates: list[str]) -> list[ScoredCandidate]:
        scored = [self.score_candidate(query, candidate, original_index=index) for index, candidate in enumerate(candidates)]
        return sorted(scored, key=lambda item: (-item.score, item.original_index))

    def rerank(self, query: str, candidates: list[str]) -> list[str]:
        return [item.candidate for item in self.score_candidates(query, candidates)]
