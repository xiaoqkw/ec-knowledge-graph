import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pydantic import BaseModel

from agent.runtime import AgentRuntime, TraceStore


class EchoInput(BaseModel):
    value: str


class EchoOutput(BaseModel):
    value: str


class EchoTool:
    name = "echo_tool"
    input_model = EchoInput
    output_model = EchoOutput
    failure_modes = ("tool_error",)
    latency_budget_ms = 100

    def run(self, payload: EchoInput) -> EchoOutput:
        return EchoOutput(value=payload.value)


class AgentRuntimeTestCase(unittest.TestCase):
    def test_tool_only_trace_can_be_loaded(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime = AgentRuntime(trace_store=TraceStore(Path(tmp_dir)))
            runtime.register_tool(EchoTool())
            record = runtime.run_tool_only(
                tool_name="echo_tool",
                payload={"value": "hello"},
                user_query="hi",
                session_id="session-1",
            )
            self.assertTrue(record.ok)
            trace_files = list(Path(tmp_dir).glob("*/*.json"))
            self.assertEqual(len(trace_files), 1)
            payload = json.loads(trace_files[0].read_text(encoding="utf-8"))
            trace = runtime.get_trace(payload["request_id"])
            self.assertIsNotNone(trace)
            self.assertEqual(trace.session_id, "session-1")


if __name__ == "__main__":
    unittest.main()
