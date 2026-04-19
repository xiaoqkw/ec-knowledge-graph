import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

CURRENT_DIR = Path(__file__).resolve()
SRC_DIR = CURRENT_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from configuration.config import BEST_MODEL_DIR  
from datasync.utils import MysqlReader, Neo4jWriter  
from ner.predict import Predictor, load_predictor  

class TextSynchronizer:
    """把 SPU 描述文本中抽取出的 Tag 同步到 Neo4j。"""

    def __init__(self):
        self.reader = MysqlReader()
        self.writer = Neo4jWriter()
        self.extractor = self._init_extractor()

    def close(self):
        self.reader.close()
        self.writer.close()

    def _init_extractor(self):
        """加载训练好的最佳模型，并复用 Predictor。"""
        return load_predictor()

    def sync_tag(self):
        sql = """
              SELECT id, description
              FROM spu_info
              WHERE description IS NOT NULL
                AND description <> ''
              """
        spu_desc = self.reader.read(sql)

        spu_ids = [item["id"] for item in spu_desc]
        descriptions = [item["description"] for item in spu_desc]
        tags_list = self.extractor.extract(descriptions)

        tag_properties = []
        relations = []
        for spu_id, tags in zip(spu_ids, tags_list):
            # 同一个 SPU 内先去重，避免重复标签造成重复节点和重复关系。
            seen_tags = set()
            for tag in tags:
                cleaned_tag = tag.strip()
                if not cleaned_tag or cleaned_tag in seen_tags:
                    continue
                seen_tags.add(cleaned_tag)

                # 这里用“SPU ID + 标签文本”生成稳定主键，方便重复同步时保持幂等。
                tag_id = f"{spu_id}::{cleaned_tag}"
                tag_properties.append({"id": tag_id, "name": cleaned_tag})
                relations.append({"start_id": spu_id, "end_id": tag_id})

        self.writer.create_constraints(["Tag"])
        self.writer.clear_spu_tag_relations(spu_ids)
        self.writer.write_nodes("Tag", tag_properties)
        self.writer.write_relations("Have", "SPU", "Tag", relations)

if __name__ == "__main__":
    synchronizer = TextSynchronizer()
    try:
        synchronizer.sync_tag()
    finally:
        synchronizer.close()