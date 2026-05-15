from __future__ import annotations

import json
import os
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

CURRENT_DIR = Path(__file__).resolve()
SRC_DIR = CURRENT_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from agent.types import AgentRunResult, ExecutionTrace, ToolCallRecord
from configuration.config import (  # noqa: E402
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    EMBEDDING_MODEL_NAME,
    ENTITY_INDEX_CONFIG,
    LOG_DIR,
    NEO4J_CONFIG,
)
from datasync.utils import Neo4jWriter  # noqa: E402

try:
    from json_repair import repair_json
except ImportError:
    repair_json = None

try:
    from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
    from langchain_deepseek import ChatDeepSeek
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_neo4j import Neo4jGraph, Neo4jVector
    from neo4j_graphrag.types import SearchType
except ImportError:
    JsonOutputParser = None
    StrOutputParser = None
    ChatDeepSeek = None
    HuggingFaceEmbeddings = None
    Neo4jGraph = None
    Neo4jVector = None
    SearchType = None


def dump_model(model: Any) -> Any:
    if isinstance(model, BaseModel):
        if hasattr(model, "model_dump"):
            return model.model_dump()
        return model.dict()
    return model


def parse_model(model_cls, payload):
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(payload)
    return model_cls.parse_obj(payload)


class AgentResources:
    def __init__(self, *, llm_enabled: bool = True):
        self.llm_enabled = llm_enabled
        self.json_parser = JsonOutputParser() if JsonOutputParser is not None else None
        self.str_parser = StrOutputParser() if StrOutputParser is not None else None
        self._graph = None
        self._writer = None
        self._llm = None
        self._embedding_model = None
        self._vectors: dict[str, Any] = {}

    def get_graph(self):
        if self._graph is None:
            if Neo4jGraph is None:
                raise ImportError("Knowledge graph QA dependencies are not installed in the current environment.")
            self._graph = Neo4jGraph(
                url=NEO4J_CONFIG["uri"],
                username=NEO4J_CONFIG["auth"][0],
                password=NEO4J_CONFIG["auth"][1],
            )
        return self._graph

    def get_writer(self) -> Neo4jWriter:
        if self._writer is None:
            self._writer = Neo4jWriter()
        return self._writer

    def get_schema(self) -> str:
        return self.get_graph().schema

    def get_llm(self):
        if not self.llm_enabled:
            return None
        if self._llm is None:
            if ChatDeepSeek is None:
                raise ImportError("LLM dependencies are not installed in the current environment.")
            if not DEEPSEEK_API_KEY:
                raise ValueError("DEEPSEEK_API_KEY is not configured in the current environment.")
            self._llm = ChatDeepSeek(model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY)
        return self._llm

    def get_embedding_model(self):
        if self._embedding_model is None:
            if HuggingFaceEmbeddings is None:
                raise ImportError("Embedding dependencies are not installed in the current environment.")
            self._embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embedding_model

    def get_vector(self, label: str):
        if label not in self._vectors:
            if Neo4jVector is None or SearchType is None:
                raise ImportError("Vector retrieval dependencies are not installed in the current environment.")
            index_info = ENTITY_INDEX_CONFIG[label]
            self._vectors[label] = Neo4jVector.from_existing_index(
                self.get_embedding_model(),
                url=NEO4J_CONFIG["uri"],
                username=NEO4J_CONFIG["auth"][0],
                password=NEO4J_CONFIG["auth"][1],
                index_name=index_info["vector_index"],
                keyword_index_name=index_info["fulltext_index"],
                search_type=SearchType.HYBRID,
            )
        return self._vectors[label]

    def close(self) -> None:
        graph = self._graph
        if graph is not None:
            driver = getattr(graph, "_driver", None)
            if driver is not None:
                driver.close()
        if self._writer is not None:
            self._writer.close()


class TraceStore:
    def __init__(self, root_dir: Path | None = None):
        self.root_dir = Path(root_dir or (LOG_DIR / "traces"))

    def save(self, trace: ExecutionTrace) -> None:
        trace_date = datetime.now(UTC).strftime("%Y-%m-%d")
        target_dir = self.root_dir / trace_date
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{trace.request_id}.json"
        target_path.write_text(json.dumps(dump_model(trace), ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, trace_id: str) -> ExecutionTrace | None:
        for path in self.root_dir.glob("*/*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("request_id") == trace_id:
                return parse_model(ExecutionTrace, payload)
        return None

    def list_by_session(self, session_id: str) -> list[ExecutionTrace]:
        traces = []
        for path in sorted(self.root_dir.glob("*/*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("session_id") == session_id:
                traces.append(parse_model(ExecutionTrace, payload))
        return traces


class AgentRuntime:
    def __init__(self, *, resources: AgentResources | None = None, trace_store: TraceStore | None = None):
        self.resources = resources or AgentResources()
        self.trace_store = trace_store or TraceStore()
        self.tools: dict[str, Any] = {}

    def register_tool(self, tool) -> None:
        self.tools[tool.name] = tool

    def build_trace(self, *, session_id: str | None, user_query: str, intent: str) -> ExecutionTrace:
        request_id = f"trc_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        return ExecutionTrace(
            request_id=request_id,
            session_id=session_id or uuid.uuid4().hex,
            user_query=user_query,
            intent=intent,
        )

    def invoke_tool(self, trace: ExecutionTrace, tool_name: str, payload: dict[str, Any]) -> ToolCallRecord:
        tool = self.tools[tool_name]
        start = time.perf_counter()
        try:
            parsed_payload = parse_model(tool.input_model, payload)
        except ValidationError as exc:
            record = ToolCallRecord(
                tool_name=tool_name,
                input_payload=payload,
                ok=False,
                failure_mode="input_schema_invalid",
                latency_ms=0,
                metadata={"error": str(exc)},
            )
            trace.tool_calls.append(record)
            return record

        try:
            output = tool.run(parsed_payload)
            latency_ms = int((time.perf_counter() - start) * 1000)
            record = ToolCallRecord(
                tool_name=tool_name,
                input_payload=dump_model(parsed_payload),
                ok=True,
                output_payload=dump_model(output),
                latency_ms=latency_ms,
            )
        except Exception as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            failure_mode = getattr(exc, "failure_mode", "tool_error")
            record = ToolCallRecord(
                tool_name=tool_name,
                input_payload=dump_model(parsed_payload),
                ok=False,
                failure_mode=failure_mode,
                latency_ms=latency_ms,
                metadata={"error": str(exc)},
            )
        trace.tool_calls.append(record)
        trace.latency_breakdown[tool_name] = trace.latency_breakdown.get(tool_name, 0) + record.latency_ms
        return record

    def finalize_trace(self, trace: ExecutionTrace, *, answer: str) -> AgentRunResult:
        trace.final_answer = answer
        trace.total_latency_ms = sum(trace.latency_breakdown.values())
        self.trace_store.save(trace)
        return AgentRunResult(answer=answer, session_id=trace.session_id, trace=trace)

    def run_tool_only(
        self,
        *,
        tool_name: str,
        payload: dict[str, Any],
        user_query: str,
        session_id: str | None = None,
        intent: str = "dialogue_tool",
    ) -> ToolCallRecord:
        trace = self.build_trace(session_id=session_id, user_query=user_query, intent=intent)
        trace.metadata["tool_only"] = True
        trace.plan = []
        record = self.invoke_tool(trace, tool_name, payload)
        trace.total_latency_ms = sum(trace.latency_breakdown.values())
        self.trace_store.save(trace)
        return record

    def get_trace(self, trace_id: str) -> ExecutionTrace | None:
        return self.trace_store.load(trace_id)

    def list_traces(self, session_id: str) -> list[ExecutionTrace]:
        return self.trace_store.list_by_session(session_id)

    def replay_traces(self, trace_ids: list[str]) -> list[ExecutionTrace]:
        traces = []
        for trace_id in trace_ids:
            trace = self.get_trace(trace_id)
            if trace is not None:
                traces.append(trace)
        return traces

    def close(self) -> None:
        self.resources.close()
