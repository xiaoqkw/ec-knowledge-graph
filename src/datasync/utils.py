import pymysql
from neo4j import GraphDatabase
from pymysql.cursors import DictCursor

from configuration.config import GRAPH_NODE_LABELS, MYSQL_CONFIG, NEO4J_CONFIG  

class MysqlReader:
    def __init__(self):
        self.connection = pymysql.connect(**MYSQL_CONFIG)
        self.cursor = self.connection.cursor(DictCursor)

    def read(self, sql: str):
        self.cursor.execute(sql)
        return self.cursor.fetchall()

    def close(self):
        self.cursor.close()
        self.connection.close()

class Neo4jWriter:
    def __init__(self):
        self.driver = GraphDatabase.driver(**NEO4J_CONFIG)

    def close(self):
        self.driver.close()

    def run_query(self, cypher: str, **parameters):
        return self.driver.execute_query(cypher, **parameters)

    def create_constraints(self, labels: list[str] | None = None):
        labels = labels or GRAPH_NODE_LABELS
        for label in labels:
            cypher = (
                f"CREATE CONSTRAINT {label.lower()}_id_unique IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.id IS UNIQUE"
            )
            self.run_query(cypher)

    def write_nodes(self, label: str, properties: list[dict]):
        if not properties:
            return

        cypher = f"""
            UNWIND $batch AS item
            MERGE (n:{label} {{id: item.id}})
            SET n += item
        """
        self.run_query(cypher, batch=properties)

    def write_relations(self, type: str, start_label, end_label, relations: list[dict]):
        if not relations:
            return
        cypher = f"""
            UNWIND $batch AS item
            MATCH (start:{start_label} {{id: item.start_id}})
            MATCH (end:{end_label} {{id: item.end_id}})
            MERGE (start)-[:{type}]->(end)
        """
        self.run_query(cypher, batch=relations)

    def clear_spu_relations(self, spu_ids: list[int], end_labels: list[str]):
        if not spu_ids:
            return
        if not end_labels:
            return

        cypher = """
            MATCH (s:SPU)-[r:Have]->(t)
            WHERE s.id IN $spu_ids
              AND any(label IN labels(t) WHERE label IN $end_labels)
            DELETE r
            WITH DISTINCT t
            WHERE NOT (t)<-[:Have]-(:SPU)
            DELETE t
        """
        self.run_query(cypher, spu_ids=spu_ids, end_labels=end_labels)

    def clear_spu_tag_relations(self, spu_ids: list[int]):
        self.clear_spu_relations(spu_ids, ["Tag"])
