import os
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve()
SRC_DIR = CURRENT_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from configuration.config import TEXT_ENTITY_NODE_LABELS
from datasync.utils import MysqlReader, Neo4jWriter
from ner.normalization import EntityNormalizer
from ner.predict import load_predictor


class TextSynchronizer:
    """Sync typed text entities extracted from SPU descriptions to Neo4j."""

    def __init__(self):
        self.reader = MysqlReader()
        self.writer = Neo4jWriter()
        self.extractor = load_predictor()
        self.normalizer = EntityNormalizer()

    def close(self):
        self.reader.close()
        self.writer.close()

    def sync_entities(self):
        sql = """
              SELECT id, description
              FROM spu_info
              WHERE description IS NOT NULL
                AND description <> ''
              """
        spu_desc = self.reader.read(sql)

        spu_ids = [item["id"] for item in spu_desc]
        descriptions = [item["description"] for item in spu_desc]
        extracted_entities = self.extractor.extract(descriptions)

        nodes_by_label = {label: [] for label in TEXT_ENTITY_NODE_LABELS.values()}
        relations_by_label = {label: [] for label in TEXT_ENTITY_NODE_LABELS.values()}

        for spu_id, entities in zip(spu_ids, extracted_entities):
            normalized_entities = self.normalizer.normalize_entities(entities)
            seen_entities = set()

            for entity in normalized_entities:
                canonical_name = entity["canonical_name"]
                node_label = entity["node_label"]
                dedupe_key = (node_label, canonical_name)
                if dedupe_key in seen_entities:
                    continue
                seen_entities.add(dedupe_key)

                node_id = f"{entity['entity_type']}::{canonical_name}"
                nodes_by_label[node_label].append(
                    {
                        "id": node_id,
                        "name": canonical_name,
                        "entity_type": entity["entity_type"],
                    }
                )
                relations_by_label[node_label].append(
                    {
                        "start_id": spu_id,
                        "end_id": node_id,
                    }
                )

        typed_labels = list(TEXT_ENTITY_NODE_LABELS.values())
        self.writer.create_constraints(typed_labels)
        self.writer.clear_spu_relations(spu_ids, typed_labels)

        for label in typed_labels:
            self.writer.write_nodes(label, nodes_by_label[label])
            self.writer.write_relations("Have", "SPU", label, relations_by_label[label])


if __name__ == "__main__":
    synchronizer = TextSynchronizer()
    try:
        synchronizer.sync_entities()
    finally:
        synchronizer.close()
