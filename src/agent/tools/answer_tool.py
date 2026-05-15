from __future__ import annotations

import json

from pydantic import BaseModel, Field

from agent.tools.base import ToolBase


class AnswerInput(BaseModel):
    question: str
    query_result: list[dict] = Field(default_factory=list)
    history: list[dict[str, str]] = Field(default_factory=list)


class AnswerOutput(BaseModel):
    answer: str
    llm_used: bool


class AnswerTool(ToolBase):
    name = "answer_tool"
    description = "Generate a grounded natural language answer from query results."
    input_model = AnswerInput
    output_model = AnswerOutput
    failure_modes = ("tool_error",)
    latency_budget_ms = 1500

    def __init__(self, resources):
        self.resources = resources

    def run(self, payload: AnswerInput) -> AnswerOutput:
        llm = self.resources.get_llm()
        if llm is None:
            if payload.query_result:
                return AnswerOutput(answer=json.dumps(payload.query_result, ensure_ascii=False), llm_used=False)
            return AnswerOutput(answer="当前图谱中没有找到相关信息。", llm_used=False)

        prompt = """
你是一名电商智能客服。请根据用户当前问题、最近对话历史和知识图谱查询结果，生成一段简洁、准确、自然的中文回答。
最近对话历史（可能为空）：
{history_text}

当前用户问题：{question}

查询结果：{query_result}

要求：
1. 如果查询结果为空，明确告诉用户当前图谱中没有找到相关信息。
2. 不要编造查询结果中不存在的事实。
3. 回答尽量简洁。
"""
        rendered_prompt = prompt.format(
            question=payload.question,
            history_text=self._format_history(payload.history),
            query_result=json.dumps(payload.query_result, ensure_ascii=False),
        )
        output = llm.invoke(rendered_prompt)
        content = getattr(output, "content", str(output)).strip()
        return AnswerOutput(answer=content or "当前图谱中没有找到相关信息。", llm_used=True)

    @staticmethod
    def _format_history(history: list[dict[str, str]] | None) -> str:
        if not history:
            return "无"
        lines = []
        for turn in history:
            user_message = turn.get("user", "").strip()
            assistant_message = turn.get("assistant", "").strip()
            if user_message:
                lines.append(f"用户: {user_message}")
            if assistant_message:
                lines.append(f"助手: {assistant_message}")
        return "\n".join(lines) if lines else "无"
