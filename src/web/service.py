import json
import os
import sys
from pathlib import Path

from json_repair import repair_json
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from neo4j_graphrag.types import SearchType
from sympy.physics.vector.printing import params

CURRENT_DIR = Path(__file__).resolve()
SRC_DIR = CURRENT_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from configuration.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    EMBEDDING_MODEL_NAME,
    ENTITY_INDEX_CONFIG,
    NEO4J_CONFIG,
)


class ChatService:
    """知识图谱问答服务：问题 -> 实体对齐 -> Cypher -> 答案。"""

    def __init__(self):
        self.graph = Neo4jGraph(
            url=NEO4J_CONFIG["uri"],
            username=NEO4J_CONFIG["auth"][0],
            password=NEO4J_CONFIG["auth"][1],
        )
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True},
        )

        if not DEEPSEEK_API_KEY:
            raise ValueError("未检测到 DEEPSEEK_API_KEY，请先在 .env 中配置。")

        self.llm = ChatDeepSeek(model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY)

        # 为可对齐的实体标签初始化混合检索对象。
        self.neo4j_vectors = {
            label: self._build_vector(label, index_info)
            for label, index_info in ENTITY_INDEX_CONFIG.items()
        }

        self.json_parser = JsonOutputParser()
        self.str_parser = StrOutputParser()

    def _build_vector(self, label: str, index_info: dict) -> Neo4jVector:
        """根据标签对应的全文索引和向量索引构建混合检索器。"""
        return Neo4jVector.from_existing_index(
            self.embedding_model,
            url=NEO4J_CONFIG["uri"],
            username=NEO4J_CONFIG["auth"][0],
            password=NEO4J_CONFIG["auth"][1],
            index_name=index_info["vector_index"],
            keyword_index_name=index_info["fulltext_index"],
            search_type=SearchType.HYBRID,
        )

    def chat(self, question: str) -> str:
        """执行完整问答流程，并返回自然语言答案。"""
        result = self._generate_cypher(question)
        cypher = result["cypher_query"]
        entities_to_align = result["entities_to_align"]
        print("生成的 Cypher:", cypher)
        print("对齐前的实体列表:", entities_to_align)

        aligned_entities = self._entity_align(entities_to_align)
        print("对齐后的实体列表:", aligned_entities)

        query_result = self._execute_cypher(cypher, aligned_entities)
        print("图谱查询结果:", query_result)

        answer = self._generate_answer(question, query_result)
        print("最终回答:", answer)
        return answer

    def _generate_cypher(self, question: str) -> dict:
        """根据用户问题和图谱 schema 生成参数化 Cypher。"""
        prompt = """
你是一个专业的 Neo4j Cypher 查询生成器。

你的任务：
根据用户问题和知识图谱结构信息，生成一条参数化 Cypher 查询语句，并列出需要做实体对齐的参数。

用户问题：{question}

知识图谱结构信息：{schema_info}

要求：
1. 只输出 JSON，不要输出解释说明。
2. Cypher 中必须使用参数化占位符，例如 $param_0、$param_1。
3. entities_to_align 中列出所有需要做实体对齐的参数。
4. label 字段只能使用图谱中真实存在的节点标签。
5. 如果问题不需要实体对齐，entities_to_align 返回空列表。

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
        rendered_prompt = PromptTemplate.from_template(prompt).format(
            question=question,
            schema_info=self.graph.schema,
        )
        results = self.llm.invoke(rendered_prompt)
        content = getattr(results, "content",str(results))

        repaired = repair_json(content, ensure_ascii=False)
        return self.json_parser.invoke(repaired)

    def _entity_align(self, entities_to_align: list[dict]) -> list[dict]:
        """使用 Neo4j 混合检索，把口语化实体对齐到图谱标准实体。"""
        aligned_entities = []
        for item in entities_to_align:
            label = item["label"]
            entity = item["entity"]
            neo4j_vector = self.neo4j_vectors.get(label)
            if not neo4j_vector:
                aligned_entities.append(item)
                continue

            results = neo4j_vector.similarity_search(entity, k=1)
            if not results:
                aligned_entities.append(item)
                continue

            aligned_entity = results[0].page_content
            aligned_entities.append({
                "param_name": item["param_name"],
                "entity": aligned_entity,
                "label": label,
            })

        return aligned_entities

    def _execute_cypher(self, cypher: str, aligned_entities: list[dict]):
        """将对齐结果转换为参数字典后执行 Cypher。"""
        params = {
            item["param_name"]:item["entity"]
            for item in aligned_entities
        }

        return self.graph.query(cypher, params=params)

    def _generate_answer(self, question: str, query_result) -> str:
        """把图查询结果整理成用户可读的自然语言回答。"""
        prompt = """
你是一名电商智能客服。

请根据用户问题和知识图谱查询结果，生成一段简洁、准确、自然的中文回答。

用户问题：{question}

查询结果：{query_result}

要求：
1. 如果查询结果为空，明确告诉用户当前图谱中没有找到相关信息。
2. 不要编造查询结果中不存在的事实。
3. 回答尽量简洁。
"""
        rendered_prompt = PromptTemplate.from_template(prompt).format(
            question=question,
            query_result=json.dumps(query_result, ensure_ascii=False),
        )
        output = self.llm.invoke(rendered_prompt)
        return self.str_parser.invoke(output)

if __name__ == "__main__":
    chat_service = ChatService()
    print(chat_service.chat("Apple 都有哪些产品？"))
