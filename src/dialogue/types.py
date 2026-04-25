from dataclasses import asdict, dataclass, field
from typing import Literal


ActionType = Literal["ask_slot", "recommend", "compare", "fallback_qa", "reset"]
IntentType = Literal["recommend", "inform", "compare", "reset", "fallback_qa"]

PHONE_DOMAIN = "phone_guide"
REQUIRED_SLOTS = ("budget_max", "use_case")
OPTIONAL_SLOTS = ("brand", "storage")
SUPPORTED_STORAGE_VALUES = ("64G", "128G", "256G", "512G")
SUPPORTED_USE_CASES = ("拍照", "游戏", "续航", "性价比")


@dataclass
class RecommendationItem:
    sku_id: int
    spu_id: int
    sku_name: str
    spu_name: str
    brand: str
    price: float
    reason: str
    default_img: str = ""
    storage_options: list[str] = field(default_factory=list)
    source_text: str = ""

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload.pop("source_text", None)
        return payload


@dataclass
class DialogueState:
    session_id: str
    domain: str = PHONE_DOMAIN
    intent: IntentType = "recommend"
    slots: dict = field(
        default_factory=lambda: {
            "brand": None,
            "budget_max": None,
            "use_case": None,
            "storage": None,
        }
    )
    pending_slots: list[str] = field(default_factory=list)
    last_recommendation_spu_ids: list[int] = field(default_factory=list)
    last_recommendation_sku_ids: list[int] = field(default_factory=list)
    turn_count: int = 0
    awaiting_budget_confirmation: bool = False
    suggested_budget_min: int | None = None

    def clear(self) -> None:
        self.intent = "recommend"
        self.slots = {
            "brand": None,
            "budget_max": None,
            "use_case": None,
            "storage": None,
        }
        self.pending_slots = []
        self.last_recommendation_spu_ids = []
        self.last_recommendation_sku_ids = []
        self.turn_count = 0
        self.awaiting_budget_confirmation = False
        self.suggested_budget_min = None

    def to_view(self) -> dict:
        filled_slots = {key: value for key, value in self.slots.items() if value is not None}
        view = {
            "domain": self.domain,
            "intent": self.intent,
            "filled_slots": filled_slots,
            "pending_slots": list(self.pending_slots),
        }
        if self.awaiting_budget_confirmation and self.suggested_budget_min is not None:
            view["suggested_budget_min"] = self.suggested_budget_min
        return view


@dataclass
class NLUResult:
    intent: IntentType
    slots: dict = field(default_factory=dict)
