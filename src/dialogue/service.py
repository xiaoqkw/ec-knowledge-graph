from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Callable

from configuration.config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL
from dialogue.nlu import DialogueNLU
from dialogue.state import InMemorySessionStore
from dialogue.types import DialogueState, REQUIRED_SLOTS

try:
    from langchain_deepseek import ChatDeepSeek
except ImportError:
    ChatDeepSeek = None

if TYPE_CHECKING:
    from dialogue.retrieval import PhoneGuideRetriever


QuestionAnswerHandler = Callable[[str], str | None]
CONFIRM_BUDGET_PATTERNS = (
    "帮我筛一下",
    "那就按这个价位",
    "按这个价位",
    "继续筛",
    "那你筛一下",
)
REJECT_BUDGET_PATTERNS = (
    "不可以",
    "不行",
    "不用",
    "不要",
    "算了",
    "太贵",
    "超预算",
)
SIMPLE_CONFIRM_BUDGET_PATTERNS = {"可以", "行"}
TRAILING_PUNCTUATION = "，。！？,.!?~ "


class DialogueService:
    def __init__(
        self,
        *,
        store: InMemorySessionStore | None = None,
        nlu: DialogueNLU | None = None,
        retriever: "PhoneGuideRetriever" | None = None,
        llm_enabled: bool = True,
    ):
        self.store = store or InMemorySessionStore()
        self.nlu = nlu or DialogueNLU()
        if retriever is None:
            from dialogue.retrieval import PhoneGuideRetriever

            retriever = PhoneGuideRetriever()
        self.retriever = retriever
        self.llm = None
        if llm_enabled and DEEPSEEK_API_KEY and ChatDeepSeek is not None:
            self.llm = ChatDeepSeek(model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY)

    def close(self) -> None:
        self.retriever.close()

    def chat(
        self,
        message: str,
        session_id: str | None = None,
        qa_handler: QuestionAnswerHandler | None = None,
    ) -> dict:
        state = self.store.get_or_create(session_id)
        nlu_result = self.nlu.parse(
            message,
            brand_vocabulary=self.retriever.load_brand_vocabulary(),
            state_has_context=state.turn_count > 0,
        )

        if nlu_result.intent == "reset":
            self.store.reset(state)
            return self._build_response(
                state,
                action="reset",
                message="好的，已经清空当前导购条件。你可以重新告诉我预算和需求。",
            )

        if nlu_result.intent == "fallback_qa":
            answer = qa_handler(message) if qa_handler is not None else None
            fallback_message = answer or "当前环境没有启用知识问答模式，你可以直接告诉我手机预算、品牌或用途。"
            return self._build_response(
                state,
                mode="qa_fallback",
                action="fallback_qa",
                message=fallback_message,
            )

        state.turn_count += 1
        state.intent = nlu_result.intent
        state.slots.update(nlu_result.slots)
        state.pending_slots = self._get_missing_required_slots(state)
        self.store.save(state)

        if nlu_result.intent == "compare":
            return self._handle_compare(state)

        if self._should_accept_budget_suggestion(state, message, nlu_result):
            return self._apply_budget_suggestion(state)

        if state.pending_slots:
            return self._handle_slot_question(state)

        return self._handle_recommend(state)

    @staticmethod
    def _get_missing_required_slots(state: DialogueState) -> list[str]:
        return [slot for slot in REQUIRED_SLOTS if state.slots.get(slot) is None]

    def _handle_slot_question(self, state: DialogueState) -> dict:
        next_slot = state.pending_slots[0]
        prompts = {
            "budget_max": "你的预算大概是多少？比如 3000 以内、5000 左右都可以。",
            "use_case": "你更看重哪一方面？我这边先支持拍照、游戏、续航、性价比四种诉求。",
        }
        return self._build_response(
            state,
            action="ask_slot",
            message=prompts.get(next_slot, "再补充一点购机需求，我就能开始推荐了。"),
        )

    def _handle_recommend(self, state: DialogueState) -> dict:
        state.awaiting_budget_confirmation = False
        state.suggested_budget_min = None

        recommendations = self.retriever.search(state.slots)
        if not recommendations and state.slots.get("storage"):
            original_storage = state.slots["storage"]
            relaxed_slots = dict(state.slots)
            relaxed_slots["storage"] = None
            recommendations = self.retriever.search(relaxed_slots)
            if recommendations:
                message = (
                    f"没有找到完全满足 {original_storage} 存储条件的机型，"
                    "我先放宽到同品牌同预算候选。"
                )
                state.last_recommendation_spu_ids = [item.spu_id for item in recommendations]
                state.last_recommendation_sku_ids = [item.sku_id for item in recommendations]
                self.store.save(state)
                return self._build_response(
                    state,
                    action="recommend",
                    message=self._render_response(
                        fallback=message,
                        context={
                            "kind": "relax_storage",
                            "slots": state.slots,
                            "recommendation_count": len(recommendations),
                        },
                    ),
                    recommendations=recommendations,
                )

        if not recommendations:
            minimum = self.retriever.get_min_price(state.slots.get("brand"))
            if minimum is not None and state.slots.get("budget_max") is not None and minimum > state.slots["budget_max"]:
                state.awaiting_budget_confirmation = True
                state.suggested_budget_min = minimum
            self.store.save(state)
            return self._build_response(
                state,
                action="ask_slot",
                message=self._build_no_result_message(state, minimum),
            )

        state.last_recommendation_spu_ids = [item.spu_id for item in recommendations]
        state.last_recommendation_sku_ids = [item.sku_id for item in recommendations]
        self.store.save(state)
        return self._build_response(
            state,
            action="recommend",
            message=self._render_response(
                fallback="我先按你当前的条件筛出几款在售候选，你可以继续补充品牌或存储要求。",
                context={
                    "kind": "recommend",
                    "slots": state.slots,
                    "recommendation_count": len(recommendations),
                },
            ),
            recommendations=recommendations,
        )

    def _handle_compare(self, state: DialogueState) -> dict:
        if len(state.last_recommendation_spu_ids) < 2:
            if state.pending_slots:
                return self._handle_slot_question(state)
            return self._build_response(
                state,
                action="ask_slot",
                message="当前候选不足两款，先让我给你推荐至少两款手机，再帮你做对比。",
            )

        message = self.retriever.compare(
            state.last_recommendation_spu_ids[:2],
            state.slots.get("use_case"),
        )
        return self._build_response(
            state,
            action="compare",
            message=self._render_response(
                fallback=message,
                context={
                    "kind": "compare",
                    "slots": state.slots,
                },
            ),
        )

    def _should_accept_budget_suggestion(self, state: DialogueState, message: str, nlu_result) -> bool:
        if not state.awaiting_budget_confirmation or state.suggested_budget_min is None:
            return False
        normalized = message.strip()
        if any(pattern in normalized for pattern in REJECT_BUDGET_PATTERNS):
            return False
        if "budget_max" in nlu_result.slots:
            return False
        if any(pattern in normalized for pattern in CONFIRM_BUDGET_PATTERNS):
            return True

        simple_reply = re.sub(rf"[{re.escape(TRAILING_PUNCTUATION)}]+$", "", normalized)
        return simple_reply in SIMPLE_CONFIRM_BUDGET_PATTERNS

    def _apply_budget_suggestion(self, state: DialogueState) -> dict:
        state.slots["budget_max"] = state.suggested_budget_min
        state.awaiting_budget_confirmation = False
        self.store.save(state)
        return self._handle_recommend(state)

    def _build_no_result_message(self, state: DialogueState, minimum: int | None) -> str:
        brand = state.slots.get("brand")
        budget_max = state.slots.get("budget_max")
        storage = state.slots.get("storage")
        use_case = state.slots.get("use_case")

        if brand and minimum is not None and budget_max is not None and minimum > budget_max:
            fallback = (
                f"你当前想要 {brand}，预算是 {budget_max} 元，"
                f"但当前在售的 {brand} 手机最低大约 {minimum} 元。"
                "如果你愿意，我可以直接按这个价位往上继续帮你筛。"
            )
        elif minimum is not None:
            fallback = (
                f"当前在售手机的最低价大约是 {minimum} 元。"
                "你可以提高预算，或者换一个品牌继续筛。"
            )
        else:
            fallback = "当前图谱里没有找到可推荐的在售手机。"

        return self._render_response(
            fallback=fallback,
            context={
                "kind": "no_result",
                "slots": {
                    "brand": brand,
                    "budget_max": budget_max,
                    "storage": storage,
                    "use_case": use_case,
                },
                "minimum": minimum,
                "awaiting_budget_confirmation": state.awaiting_budget_confirmation,
            },
        )

    def _render_response(self, *, fallback: str, context: dict) -> str:
        if self.llm is None:
            return fallback

        prompt = (
            "你是电商手机导购助手。"
            "请根据给定上下文，把回复润色成自然、简洁、具体的中文对话。"
            "不要编造商品和价格，不要改变原有结论。"
            "如果用户可以确认放宽预算，就明确告诉用户可以直接回复“帮我筛一下吧”。\n"
            f"上下文: {json.dumps(context, ensure_ascii=False)}\n"
            f"基础回复: {fallback}"
        )
        try:
            result = self.llm.invoke(prompt)
            content = getattr(result, "content", str(result)).strip()
        except Exception:
            return fallback
        return content or fallback

    @staticmethod
    def _build_response(
        state: DialogueState,
        *,
        mode: str = "dialogue",
        action: str,
        message: str,
        recommendations: list | None = None,
    ) -> dict:
        return {
            "session_id": state.session_id,
            "message": message,
            "mode": mode,
            "action": action,
            "state": state.to_view(),
            "recommendations": [item.to_dict() for item in recommendations or []],
        }
