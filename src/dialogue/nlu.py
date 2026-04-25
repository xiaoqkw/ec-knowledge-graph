import json
import re
from collections import OrderedDict

from configuration.config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL
from dialogue.types import NLUResult, SUPPORTED_STORAGE_VALUES, SUPPORTED_USE_CASES

try:
    from json_repair import repair_json
except ImportError:
    repair_json = None

try:
    from langchain_deepseek import ChatDeepSeek
except ImportError:
    ChatDeepSeek = None


USE_CASE_KEYWORDS = {
    "拍照": ("拍照", "影像", "自拍", "镜头", "摄影"),
    "游戏": ("游戏", "打游戏", "电竞", "高刷", "性能"),
    "续航": ("续航", "电池", "快充", "充电", "耐用"),
    "性价比": ("性价比", "划算", "实惠", "便宜", "预算有限"),
}
BRAND_ALIASES = {
    "apple": "苹果",
    "iphone": "苹果",
    "苹果": "苹果",
    "redmi": "红米",
    "红米": "红米",
    "xiaomi": "小米",
    "小米": "小米",
    "huawei": "华为",
    "华为": "华为",
    "oppo": "OPPO",
    "vivo": "VIVO",
}
COMPARE_PATTERNS = ("比一下", "比较", "对比", "哪个好", "哪个更好")
RESET_PATTERNS = ("重新开始", "重置", "清空条件", "重新选", "从头开始")
PHONE_PATTERNS = ("手机", "买", "推荐", "选", "购机", "预算")
STORAGE_PATTERN = re.compile(r"(?<!\d)(64|128|256|512)\s*g(?:b)?(?!\w)", re.IGNORECASE)
BUDGET_PATTERN = re.compile(r"(?P<amount>\d+(?:\.\d+)?)\s*(?P<unit>k|K|千|元|块)?")


class DialogueNLU:
    def __init__(self, llm_enabled: bool = True):
        self.llm = None
        if llm_enabled and DEEPSEEK_API_KEY and ChatDeepSeek is not None:
            self.llm = ChatDeepSeek(model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY)

    def parse(
        self,
        message: str,
        *,
        brand_vocabulary: list[str],
        state_has_context: bool,
    ) -> NLUResult:
        stripped = message.strip()
        if not stripped:
            return NLUResult(intent="fallback_qa")

        rules_result = self._parse_rules(
            stripped,
            brand_vocabulary=brand_vocabulary,
            state_has_context=state_has_context,
        )
        if self._is_confident(rules_result, state_has_context):
            return rules_result

        llm_result = self._parse_with_llm(stripped, brand_vocabulary)
        if llm_result is not None:
            return llm_result
        return rules_result

    def _parse_rules(
        self,
        message: str,
        *,
        brand_vocabulary: list[str],
        state_has_context: bool,
    ) -> NLUResult:
        normalized = message.lower()
        has_phone_context = any(token in message for token in PHONE_PATTERNS) or state_has_context

        if any(pattern in message for pattern in RESET_PATTERNS):
            return NLUResult(intent="reset")

        is_compare = any(pattern in message for pattern in COMPARE_PATTERNS)

        slots = OrderedDict()
        brand = self._extract_brand(normalized, brand_vocabulary)
        if brand is not None:
            slots["brand"] = brand

        storage = self._extract_storage(normalized)
        if storage is not None:
            slots["storage"] = storage

        budget_max = self._extract_budget(
            message,
            has_phone_context=has_phone_context,
            storage=storage,
        )
        if budget_max is not None:
            slots["budget_max"] = budget_max

        use_case = self._extract_use_case(message)
        if use_case is not None:
            slots["use_case"] = use_case

        if is_compare:
            return NLUResult(intent="compare", slots=dict(slots))

        if slots:
            if not has_phone_context and set(slots.keys()) == {"brand"}:
                return NLUResult(intent="fallback_qa")
            intent = "recommend" if any(token in message for token in PHONE_PATTERNS) else "inform"
            return NLUResult(intent=intent, slots=dict(slots))

        if has_phone_context:
            return NLUResult(intent="recommend")
        return NLUResult(intent="fallback_qa")

    @staticmethod
    def _is_confident(result: NLUResult, state_has_context: bool) -> bool:
        if result.intent in {"reset", "compare"}:
            return True
        if result.slots:
            return True
        return result.intent == "recommend" and state_has_context

    @staticmethod
    def _extract_brand(normalized_message: str, brand_vocabulary: list[str]) -> str | None:
        canonical_map = {}
        for brand in brand_vocabulary:
            canonical_map[brand.lower()] = brand
        for alias, canonical in BRAND_ALIASES.items():
            if canonical in brand_vocabulary:
                canonical_map.setdefault(alias.lower(), canonical)

        for alias, canonical in canonical_map.items():
            if alias in normalized_message:
                return canonical
        return None

    @staticmethod
    def _extract_budget(
        message: str,
        *,
        has_phone_context: bool,
        storage: str | None,
    ) -> int | None:
        budget_message = message
        if storage is not None:
            budget_message = STORAGE_PATTERN.sub(" ", budget_message)

        if "预算" not in budget_message and not re.search(r"\d", budget_message):
            return None
        if storage is not None and re.fullmatch(r"\s*\d+\s*g(?:b)?\s*", message.strip(), re.IGNORECASE):
            return None

        numbers = []
        for match in BUDGET_PATTERN.finditer(budget_message):
            amount = float(match.group("amount"))
            unit = match.group("unit")
            if unit in {"k", "K", "千"}:
                amount *= 1000
            numbers.append((int(amount), unit))

        if not numbers:
            return None

        values = [value for value, _ in numbers]
        if "左右" in budget_message:
            return values[0]
        if any(token in budget_message for token in ("以内", "以下", "不超过", "最多")):
            return max(values)
        if len(values) >= 2 and any(token in budget_message for token in ("-", "~", "到", "至")):
            return max(values[:2])
        if "预算" in budget_message or "元" in budget_message or "块" in budget_message:
            return max(values)

        if has_phone_context:
            explicit_k = any(unit in {"k", "K", "千"} for _, unit in numbers)
            if explicit_k:
                return max(values)
            large_numbers = [value for value in values if value >= 100]
            if large_numbers:
                return max(large_numbers)
        return None

    @staticmethod
    def _extract_use_case(message: str) -> str | None:
        for use_case, keywords in USE_CASE_KEYWORDS.items():
            if any(keyword in message for keyword in keywords):
                return use_case
        return None

    @staticmethod
    def _extract_storage(normalized_message: str) -> str | None:
        match = STORAGE_PATTERN.search(normalized_message)
        if not match:
            return None
        storage = f"{match.group(1)}G"
        if storage in SUPPORTED_STORAGE_VALUES:
            return storage
        return None

    def _parse_with_llm(self, message: str, brand_vocabulary: list[str]) -> NLUResult | None:
        if self.llm is None:
            return None

        prompt = f"""
你是电商手机导购系统的 NLU 组件。
请只输出 JSON，不要输出解释。

允许的 intent:
- recommend
- inform
- compare
- reset
- fallback_qa

允许的 use_case:
{json.dumps(SUPPORTED_USE_CASES, ensure_ascii=False)}

允许的 storage:
{json.dumps(SUPPORTED_STORAGE_VALUES, ensure_ascii=False)}

允许的 brand:
{json.dumps(brand_vocabulary, ensure_ascii=False)}

请从用户输入中抽取:
- intent
- brand
- budget_max
- use_case
- storage

用户输入: {message}

输出格式:
{{
  "intent": "recommend",
  "slots": {{
    "brand": null,
    "budget_max": null,
    "use_case": null,
    "storage": null
  }}
}}
"""

        try:
            result = self.llm.invoke(prompt)
            content = getattr(result, "content", str(result))
            repaired = repair_json(content, ensure_ascii=False) if repair_json is not None else content
            payload = json.loads(repaired)
        except Exception:
            return None

        intent = payload.get("intent")
        slots = payload.get("slots") or {}
        if intent not in {"recommend", "inform", "compare", "reset", "fallback_qa"}:
            return None

        filtered_slots = {}
        brand = slots.get("brand")
        if brand in brand_vocabulary:
            filtered_slots["brand"] = brand

        budget_max = slots.get("budget_max")
        if isinstance(budget_max, (int, float)) and budget_max > 0:
            filtered_slots["budget_max"] = int(budget_max)

        use_case = slots.get("use_case")
        if use_case in SUPPORTED_USE_CASES:
            filtered_slots["use_case"] = use_case

        storage = slots.get("storage")
        if storage in SUPPORTED_STORAGE_VALUES:
            filtered_slots["storage"] = storage

        return NLUResult(intent=intent, slots=filtered_slots)
