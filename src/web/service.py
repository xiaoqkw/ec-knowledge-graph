import json
import os
import re
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve()
SRC_DIR = CURRENT_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from agent import AgentController, AgentRuntime, AgentResources
from agent.runtime import dump_model
from configuration.config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL, EMBEDDING_MODEL_NAME, ENTITY_INDEX_CONFIG, NEO4J_CONFIG

try:
    from json_repair import repair_json
except ImportError:
    repair_json = None

try:
    from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_deepseek import ChatDeepSeek
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_neo4j import Neo4jGraph, Neo4jVector
    from neo4j_graphrag.types import SearchType
except ImportError:
    JsonOutputParser = None
    StrOutputParser = None
    PromptTemplate = None
    ChatDeepSeek = None
    HuggingFaceEmbeddings = None
    Neo4jGraph = None
    Neo4jVector = None
    SearchType = None


UNSAFE_CYPHER_PATTERN = re.compile(r"\b(create|merge|set|delete|remove|drop)\b", re.IGNORECASE)


class ChatService:
    """Knowledge graph QA service compatibility layer."""

    def __init__(self, *, llm_enabled: bool = True, controller: AgentController | None = None):
        if controller is None:
            runtime = AgentRuntime(resources=AgentResources(llm_enabled=llm_enabled))
            controller = AgentController(runtime, enable_explain=True)
        self.controller = controller
        self.runtime = controller.runtime
        self.resources = self.runtime.resources
        self.graph = None
        self.embedding_model = None
        self.neo4j_vectors = {}
        self.llm = self.resources.get_llm() if llm_enabled else None
        self.json_parser = self.resources.json_parser
        self.str_parser = self.resources.str_parser

    def close(self) -> None:
        self.runtime.close()

    def chat(self, question: str, history: list[dict[str, str]] | None = None) -> str:
        return self.trace_chat(question, history=history, align_entities=True)["answer"]

    def run_agent(
        self,
        question: str,
        *,
        history: list[dict[str, str]] | None = None,
        session_id: str | None = None,
        align_entities: bool = True,
    ):
        return self.controller.run_kgqa(
            question,
            session_id=session_id,
            history=history,
            align_entities=align_entities,
        )

    def trace_chat(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        *,
        align_entities: bool = True,
    ) -> dict:
        if not hasattr(self, "controller") or self.controller is None:
            return self._trace_chat_legacy(question, history=history, align_entities=align_entities)

        result = self.run_agent(question, history=history, align_entities=align_entities)
        trace = result.trace
        metadata = trace.metadata
        execution_error = None
        for record in trace.tool_calls:
            if not record.ok and record.metadata.get("error"):
                execution_error = record.metadata["error"]
                break
        return {
            "question": question,
            "raw_cypher_output": metadata.get("planner_raw_output", ""),
            "repaired_cypher_output": metadata.get("planner_repaired_output", ""),
            "parse_success_raw": bool(metadata.get("planner_parse_success_raw", False)),
            "parse_success_repaired": bool(metadata.get("planner_parse_success_repaired", False)),
            "cypher_query_present": bool(metadata.get("cypher_query")),
            "parsed_payload": metadata.get("parsed_payload"),
            "cypher_query": metadata.get("cypher_query", ""),
            "entities_to_align": metadata.get("entities_to_align", []),
            "aligned_entities": metadata.get("aligned_entities", []),
            "executed_params": metadata.get("executed_params", {}),
            "unsafe_cypher": "unsafe_query_blocked" in trace.failure_tags,
            "execution_success": any(record.tool_name == "graph_query_tool" and record.ok for record in trace.tool_calls),
            "execution_error": execution_error,
            "query_result": metadata.get("query_result", []),
            "non_empty_result": bool(metadata.get("query_result", [])),
            "answer": result.answer,
            "trace_id": trace.request_id,
        }

    def _trace_chat_legacy(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        *,
        align_entities: bool = True,
    ) -> dict:
        raw_prompt = self._build_cypher_prompt(question, history)
        raw_cypher_output = self._invoke_cypher_llm(raw_prompt)
        parse_result = self._parse_cypher_output(raw_cypher_output)

        payload = parse_result["payload"] if isinstance(parse_result["payload"], dict) else {}
        cypher_query = payload.get("cypher_query", "") if parse_result["cypher_query_present"] else ""
        entities_to_align = self._normalize_entities_to_align(payload.get("entities_to_align"))
        aligned_entities = self._entity_align(entities_to_align) if align_entities else [dict(item) for item in entities_to_align]
        executed_params = self._build_executed_params(aligned_entities)

        unsafe_cypher = self._is_unsafe_cypher(cypher_query)
        execution_success = False
        execution_error = None
        query_result = []
        non_empty_result = False

        if parse_result["cypher_query_present"]:
            try:
                query_result = self._execute_cypher(cypher_query, executed_params)
                execution_success = True
                non_empty_result = bool(query_result)
            except Exception as exc:
                execution_error = str(exc)
        else:
            execution_error = parse_result["parse_error"] or "Missing cypher_query in parsed payload."

        answer = self._generate_answer(question, query_result, history=history)
        return {
            "question": question,
            "raw_cypher_output": parse_result["raw_content"],
            "repaired_cypher_output": parse_result["repaired_content"],
            "parse_success_raw": parse_result["parse_success_raw"],
            "parse_success_repaired": parse_result["parse_success_repaired"],
            "cypher_query_present": parse_result["cypher_query_present"],
            "parsed_payload": parse_result["payload"],
            "cypher_query": cypher_query,
            "entities_to_align": entities_to_align,
            "aligned_entities": aligned_entities,
            "executed_params": executed_params,
            "unsafe_cypher": unsafe_cypher,
            "execution_success": execution_success,
            "execution_error": execution_error,
            "query_result": query_result,
            "non_empty_result": non_empty_result,
            "answer": answer,
        }

    def _build_cypher_prompt(self, question: str, history: list[dict[str, str]] | None = None) -> str:
        prompt = """
你是一个专业的 Neo4j Cypher 查询生成器。你的任务：根据用户当前问题、最近对话历史和知识图谱结构信息，生成一条参数化 Cypher 查询语句，并列出需要做实体对齐的参数。
最近对话历史（可能为空）：
{history_text}

当前用户问题：{question}

知识图谱结构信息：{schema_info}

要求：
1. 如果当前问题出现“它 / 这个 / 那个 / 这些 / 那些”等省略指代，可以结合最近历史恢复指代对象，但不要编造历史中不存在的实体或约束。
2. 只输出 JSON，不要输出解释说明。
3. Cypher 中必须使用参数化占位符，例如 $param_0、$param_1。
4. entities_to_align 中列出所有需要做实体对齐的参数。
5. label 字段只能使用图谱中真实存在的节点标签。
6. 如果问题不需要实体对齐，entities_to_align 返回空列表。
输出格式：
{{
  "cypher_query": "生成的 Cypher 语句",
  "entities_to_align": [
    {{
      "param_name": "param_0",
      "entity": "原始实体名称",
      "label": "节点标签"
    }}
  ]
}}
"""
        graph = self._ensure_graph()
        return PromptTemplate.from_template(prompt).format(
            question=question,
            history_text=self._format_history(history),
            schema_info=graph.schema,
        )

    def _invoke_cypher_llm(self, prompt: str) -> str:
        if self.llm is None:
            raise RuntimeError("LLM is not enabled for the current ChatService instance.")
        result = self.llm.invoke(prompt)
        return getattr(result, "content", str(result))

    def _parse_cypher_output(self, raw_content: str) -> dict:
        parse_success_raw = False
        parse_success_repaired = False
        payload = None
        parse_error = None
        repaired_content = raw_content

        try:
            payload = self.json_parser.invoke(raw_content)
            parse_success_raw = True
            parse_success_repaired = True
        except Exception as raw_exc:
            parse_error = str(raw_exc)
            repaired_content = repair_json(raw_content, ensure_ascii=False) if repair_json is not None else raw_content
            try:
                payload = self.json_parser.invoke(repaired_content)
                parse_success_repaired = True
            except Exception as repaired_exc:
                parse_error = str(repaired_exc)

        if payload is not None and not isinstance(payload, dict):
            parse_error = f"Parsed JSON payload must be an object, got {type(payload).__name__}."
            payload = None
            parse_success_raw = False
            parse_success_repaired = False

        cypher_query_present = bool(payload and str(payload.get("cypher_query", "")).strip())
        return {
            "raw_content": raw_content,
            "repaired_content": repaired_content,
            "parse_success_raw": parse_success_raw,
            "parse_success_repaired": parse_success_repaired,
            "cypher_query_present": cypher_query_present,
            "payload": payload,
            "parse_error": parse_error,
        }

    def _search_entities(self, label: str, query: str, mode: str, k: int) -> list[str]:
        if hasattr(self, "runtime") and self.runtime is not None:
            record = self.runtime.run_tool_only(
                tool_name="entity_link_tool",
                payload={
                    "entities": [{"param_name": "param_0", "entity": query, "label": label}],
                    "mode": mode,
                    "top_k": k,
                },
                user_query=query,
                intent="entity_link_eval",
            )
            if not record.ok:
                return []
            candidates = record.output_payload.get("candidates_by_param", {}).get("param_0", [])
            return [item["name"] for item in candidates]

        graph = self._ensure_graph()
        if mode == "exact_match":
            rows = graph.query(
                f"""
                MATCH (n:{label} {{name: $query}})
                RETURN n.name AS name
                """,
                params={"query": query},
            )
            return self._dedupe_candidates(row.get("name") for row in rows)
        if mode == "fulltext":
            index_info = ENTITY_INDEX_CONFIG.get(label)
            if index_info is None:
                return []
            rows = graph.query(
                """
                CALL db.index.fulltext.queryNodes($index_name, $query) YIELD node, score
                RETURN node.name AS name, score
                ORDER BY score DESC
                LIMIT $limit
                """,
                params={"index_name": index_info["fulltext_index"], "query": query, "limit": k},
            )
            return self._dedupe_candidates(row.get("name") for row in rows)
        if mode == "hybrid":
            vector = self._ensure_vector(label)
            results = vector.similarity_search(query, k=k)
            return self._dedupe_candidates(item.page_content for item in results)
        raise ValueError(f"Unsupported search mode: {mode}")

    def _entity_align(self, entities_to_align: list[dict], *, mode: str = "hybrid", k: int = 1) -> list[dict]:
        if hasattr(self, "runtime") and self.runtime is not None:
            record = self.runtime.run_tool_only(
                tool_name="entity_link_tool",
                payload={"entities": entities_to_align, "mode": mode, "top_k": max(k, 1)},
                user_query="entity_align",
                intent="entity_link",
            )
            if not record.ok:
                return [dict(item) for item in entities_to_align]
            return record.output_payload.get("aligned_entities", [])

        aligned_entities = []
        for item in entities_to_align:
            candidates = self._search_entities(item["label"], item["entity"], mode=mode, k=k)
            if not candidates:
                aligned_entities.append(dict(item))
                continue
            aligned_entities.append({"param_name": item["param_name"], "entity": candidates[0], "label": item["label"]})
        return aligned_entities

    def _execute_cypher(self, cypher: str, executed_params: dict):
        if hasattr(self, "runtime") and self.runtime is not None:
            record = self.runtime.run_tool_only(
                tool_name="graph_query_tool",
                payload={"cypher_query": cypher, "params": executed_params},
                user_query="graph_query",
                intent="graph_query",
            )
            if not record.ok:
                raise RuntimeError(record.metadata.get("error", "Graph query execution failed."))
            return record.output_payload.get("rows", [])

        graph = self._ensure_graph()
        return graph.query(cypher, params=executed_params)

    def _generate_answer(self, question: str, query_result, history: list[dict[str, str]] | None = None) -> str:
        if hasattr(self, "runtime") and self.runtime is not None:
            record = self.runtime.run_tool_only(
                tool_name="answer_tool",
                payload={"question": question, "query_result": query_result, "history": history or []},
                user_query=question,
                intent="answer",
            )
            if not record.ok:
                if query_result:
                    return json.dumps(query_result, ensure_ascii=False)
                return "当前图谱中没有找到相关信息。"
            return record.output_payload.get("answer", "当前图谱中没有找到相关信息。")

        rendered_prompt = PromptTemplate.from_template(
            """
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
        ).format(
            question=question,
            history_text=self._format_history(history),
            query_result=json.dumps(query_result, ensure_ascii=False),
        )
        if self.llm is None:
            if query_result:
                return json.dumps(query_result, ensure_ascii=False)
            return "当前图谱中没有找到相关信息。"
        output = self.llm.invoke(rendered_prompt)
        return self.str_parser.invoke(output)

    @staticmethod
    def _build_executed_params(entities: list[dict]) -> dict[str, str]:
        return {item["param_name"]: item["entity"] for item in entities if item.get("param_name") and item.get("entity") is not None}

    @staticmethod
    def _dedupe_candidates(candidates) -> list[str]:
        names = []
        seen = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            names.append(candidate)
        return names

    @staticmethod
    def _normalize_entities_to_align(raw_entities) -> list[dict]:
        if not isinstance(raw_entities, list):
            return []
        normalized = []
        for item in raw_entities:
            if not isinstance(item, dict):
                continue
            param_name = str(item.get("param_name", "")).strip()
            entity = str(item.get("entity", "")).strip()
            label = str(item.get("label", "")).strip()
            if not param_name or not entity or not label:
                continue
            normalized.append({"param_name": param_name, "entity": entity, "label": label})
        return normalized

    @staticmethod
    def _is_unsafe_cypher(cypher_query: str) -> bool:
        return bool(UNSAFE_CYPHER_PATTERN.search(cypher_query or ""))

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

    def _ensure_graph(self):
        if self.graph is not None:
            return self.graph
        self.graph = Neo4jGraph(
            url=NEO4J_CONFIG["uri"],
            username=NEO4J_CONFIG["auth"][0],
            password=NEO4J_CONFIG["auth"][1],
        )
        return self.graph

    def _ensure_vector(self, label: str):
        if label in self.neo4j_vectors:
            return self.neo4j_vectors[label]
        if self.embedding_model is None:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                encode_kwargs={"normalize_embeddings": True},
            )
        index_info = ENTITY_INDEX_CONFIG[label]
        self.neo4j_vectors[label] = Neo4jVector.from_existing_index(
            self.embedding_model,
            url=NEO4J_CONFIG["uri"],
            username=NEO4J_CONFIG["auth"][0],
            password=NEO4J_CONFIG["auth"][1],
            index_name=index_info["vector_index"],
            keyword_index_name=index_info["fulltext_index"],
            search_type=SearchType.HYBRID,
        )
        return self.neo4j_vectors[label]
