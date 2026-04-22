import math
import os
import sys
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve()
SRC_DIR = CURRENT_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from configuration.config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL, TEXT_ENTITY_NODE_LABELS
from datasync.utils import Neo4jWriter
from ner.normalization import EntityNormalizer
from ner.predict import load_predictor


DEFAULT_DESC_PROMPT = """
你是一个中文电商商品描述生成助手。
请根据给定的商品标题、平台属性和销售属性，生成一条自然、简洁、适合商品图谱节点使用的中文描述。
要求：
1. 只基于输入信息生成，不要编造不存在的品牌、材质、规格或功效。
2. 保留最关键的信息：商品名称、主要属性、规格、人群或使用场景。
3. 输出 1 句话，长度控制在 30 到 80 字之间。
4. 不要输出项目符号，不要输出解释。
商品标题：{title}
平台属性：{item_pvs}
销售属性：{sku_pvs}
"""


class OpenBGTextSynchronizer:
    def __init__(self):
        self.writer = Neo4jWriter()
        self.extractor = load_predictor()
        self.normalizer = EntityNormalizer()
        self.desc_concurrency = int(os.getenv("OPENBG_DESC_CONCURRENCY", "6"))
        self.desc_batch_size = int(os.getenv("OPENBG_DESC_BATCH_SIZE", "60"))
        self.ner_batch_size = int(os.getenv("OPENBG_NER_BATCH_SIZE", "64"))
        self.desc_chain = self._build_desc_chain()

    def close(self):
        self.writer.close()

    def run(self):
        rows = self.writer.query(
            """
            MATCH (s:SPU)
            WHERE s.source = 'openbg'
            RETURN s.id AS id,
                   s.name AS title,
                   coalesce(s.item_pvs_raw, '') AS item_pvs,
                   coalesce(s.sku_pvs_raw, '') AS sku_pvs,
                   coalesce(s.description, '') AS description
            """
        )
        if not rows:
            print("[openbg_text_sync] No openbg SPU nodes found.")
            return

        print(f"[openbg_text_sync] Loaded {len(rows)} openbg SPUs.")
        descriptions = self._generate_descriptions(rows)
        spu_ids = [row["id"] for row in rows]

        for spu_id, description in tqdm(
            zip(spu_ids, descriptions),
            total=len(spu_ids),
            desc="Write descriptions",
        ):
            self.writer.run_query(
                """
                MATCH (s:SPU {id: $spu_id})
                SET s.description = $description
                """,
                spu_id=spu_id,
                description=description,
            )

        extracted_entities = self._extract_entities_in_batches(descriptions)
        nodes_by_label = {label: [] for label in TEXT_ENTITY_NODE_LABELS.values()}
        relations_by_label = {label: [] for label in TEXT_ENTITY_NODE_LABELS.values()}

        for spu_id, entities in tqdm(
            zip(spu_ids, extracted_entities),
            total=len(spu_ids),
            desc="Normalize entities",
        ):
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
        print("[openbg_text_sync] Refreshing text entity relations in Neo4j...")
        self.writer.create_constraints(typed_labels)
        self.writer.clear_spu_relations(spu_ids, typed_labels)

        for label in typed_labels:
            print(
                f"[openbg_text_sync] Writing {label}: "
                f"{len(nodes_by_label[label])} nodes, {len(relations_by_label[label])} relations"
            )
            self.writer.write_nodes(label, nodes_by_label[label])
            self.writer.write_relations("Have", "SPU", label, relations_by_label[label])

        print("[openbg_text_sync] Done.")

    def _build_desc_chain(self):
        if not DEEPSEEK_API_KEY:
            return None
        prompt = PromptTemplate.from_template(DEFAULT_DESC_PROMPT)
        llm = ChatDeepSeek(model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY)
        parser = StrOutputParser()
        return prompt | llm | parser

    def _generate_descriptions(self, rows: list[dict]) -> list[str]:
        descriptions = [""] * len(rows)
        llm_inputs = []
        llm_indices = []

        for index, row in enumerate(rows):
            existing_description = str(row["description"]).strip()
            if existing_description:
                descriptions[index] = existing_description
                continue

            llm_inputs.append(
                {
                    "title": row["title"],
                    "item_pvs": row["item_pvs"],
                    "sku_pvs": row["sku_pvs"],
                }
            )
            llm_indices.append(index)

        reused_count = len(rows) - len(llm_inputs)
        if reused_count:
            print(f"[openbg_text_sync] Reusing {reused_count} existing descriptions.")

        if self.desc_chain is not None and llm_inputs:
            total_batches = math.ceil(len(llm_inputs) / self.desc_batch_size)
            for start in tqdm(
                range(0, len(llm_inputs), self.desc_batch_size),
                total=total_batches,
                desc="Generate descriptions",
            ):
                input_batch = llm_inputs[start : start + self.desc_batch_size]
                index_batch = llm_indices[start : start + self.desc_batch_size]
                try:
                    results = self.desc_chain.batch(
                        input_batch,
                        config={"max_concurrency": self.desc_concurrency},
                        return_exceptions=True,
                    )
                except Exception:
                    results = [None] * len(input_batch)

                for row_index, result in zip(index_batch, results):
                    if isinstance(result, Exception):
                        result = ""

                    description = str(result).strip() if result is not None else ""
                    if description:
                        descriptions[row_index] = description

        fallback_count = 0
        for index, row in enumerate(rows):
            if descriptions[index]:
                continue
            descriptions[index] = self._fallback_description(
                row["title"],
                row["item_pvs"],
                row["sku_pvs"],
                row["description"],
            )
            fallback_count += 1

        if fallback_count:
            print(f"[openbg_text_sync] Used fallback descriptions for {fallback_count} items.")

        return descriptions

    def _extract_entities_in_batches(self, descriptions: list[str]) -> list[list[dict]]:
        if not descriptions:
            return []

        outputs = []
        batch_size = max(1, self.ner_batch_size)
        total_batches = math.ceil(len(descriptions) / batch_size)
        for start in tqdm(
            range(0, len(descriptions), batch_size),
            total=total_batches,
            desc="Run NER",
        ):
            batch = descriptions[start : start + batch_size]
            outputs.extend(self.extractor.extract(batch))
        return outputs

    @staticmethod
    def _fallback_description(title: str, item_pvs: str, sku_pvs: str, existing_description: str) -> str:
        if existing_description.strip():
            return existing_description.strip()

        parts = [title.strip()]
        for raw_field in (item_pvs, sku_pvs):
            for item in raw_field.split("#;#"):
                if "#:#" not in item:
                    continue
                key, value = item.split("#:#", 1)
                key = key.strip()
                value = value.strip()
                if key and value:
                    parts.append(f"{key}{value}")
                if len(parts) >= 4:
                    break
            if len(parts) >= 4:
                break
        return "，".join(part for part in parts if part)


if __name__ == "__main__":
    synchronizer = OpenBGTextSynchronizer()
    try:
        synchronizer.run()
    finally:
        synchronizer.close()
