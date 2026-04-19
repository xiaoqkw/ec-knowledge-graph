import os
import sys
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph

CURRENT_DIR = Path(__file__).resolve()
SRC_DIR = CURRENT_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from configuration.config import (
    EMBEDDING_MODEL_NAME,
    ENTITY_INDEX_CONFIG,
    NEO4J_CONFIG,
)


class IndexUtil:
    """创建全文索引、向量索引并为节点写入嵌入向量。"""

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

    def create_fulltext_index(self, index_name: str, label: str, property_name: str):
        """为节点名称创建全文检索索引。"""
        cypher = f"""
            CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
            FOR (n:{label}) ON EACH [n.{property_name}]
        """
        self.graph.query(cypher)

    def create_vector_index(
        self,
        index_name: str,
        label: str,
        source_property: str,
        embedding_property: str,
    ):
        """写入嵌入向量后，再创建 Neo4j 向量索引。"""
        embedding_dim = self._add_embedding(
            label=label,
            source_property=source_property,
            embedding_property=embedding_property,
        )

        cypher = f"""
            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
            FOR (n:{label}) ON n.{embedding_property}
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {embedding_dim},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
        """
        self.graph.query(cypher)

    def _add_embedding(
            self,
            label: str,
            source_property: str,
            embedding_property: str,
    ) -> int:
        """为指定标签的节点批量生成嵌入向量并写回图数据库。"""
        cypher = f"""
            MATCH (n:{label}) 
            RETURN n.{source_property} AS text, id(n) AS id
        """
        results = self.graph.query(cypher)
        docs = [item['text'] for item in results if item['text']]
        ids = [item['id'] for item in results if item['text']]

        embeddings = self.embedding_model.embed_documents(docs)
        batch = [
            {'id':node_id, 'embedding':embedding}
            for node_id, embedding in zip(ids, embeddings)
        ]

        cypher = f"""
            UNWIND $batch AS item
            MATCH (n:{label}) 
            WHERE id(n) = item.id
            SET n.{embedding_property} = item.embedding
        """
        self.graph.query(cypher, params={'batch':batch})
        return len(embeddings[0])

    def create_all_indexes(self):
        """为用到的全部实体标签创建全文与向量索引。"""
        for label, index_info in ENTITY_INDEX_CONFIG.items():
            self.create_fulltext_index(
                index_name=index_info["fulltext_index"],
                label=label,
                property_name="name",
            )
            self.create_vector_index(
                index_name=index_info["vector_index"],
                label=label,
                source_property="name",
                embedding_property="embedding",
            )

if __name__ == "__main__":
    util = IndexUtil()
    util.create_all_indexes()
    print("Day 04 全部索引创建完成。")
