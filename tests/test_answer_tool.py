import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent.tools.answer_tool import AnswerTool


class PromptCaptureLLM:
    def __init__(self):
        self.prompt = ""

    def invoke(self, prompt):
        self.prompt = prompt
        return type("Result", (), {"content": "ok"})()


class StubResources:
    def __init__(self, llm):
        self._llm = llm

    def get_llm(self):
        return self._llm


class AnswerToolTestCase(unittest.TestCase):
    def test_prompt_excludes_embedding_fields_from_query_result(self):
        llm = PromptCaptureLLM()
        tool = AnswerTool(StubResources(llm))
        payload = tool.input_model(
            question="这是什么产品？",
            query_result=[
                {
                    "product_name": "Apple iPhone 12",
                    "embedding": [0.1, 0.2, 0.3],
                    "nested": {
                        "embedding": [0.4, 0.5],
                        "brand": "Apple",
                    },
                }
            ],
            history=[],
            evidence=[],
        )

        result = tool.run(payload)

        self.assertEqual(result.answer, "ok")
        self.assertNotIn('"embedding"', llm.prompt)
        self.assertIn('"product_name": "Apple iPhone 12"', llm.prompt)
        self.assertIn('"brand": "Apple"', llm.prompt)


if __name__ == "__main__":
    unittest.main()
