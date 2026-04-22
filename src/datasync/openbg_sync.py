import sys
from collections import defaultdict
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve()
SRC_DIR = CURRENT_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from configuration.config import GRAPH_NODE_LABELS
from datasync.openbg_common import (
    OPENBG_JSONL_FILE,
    SKU_SALE_ATTR_WHITELIST,
    Category2Candidate,
    Category3Candidate,
    choose_best_category2,
    choose_best_category3,
    extract_brand,
    load_openbg_rows,
    normalize_text,
    parse_pvs,
    stable_id,
)
from datasync.utils import Neo4jWriter


CATEGORY3_MATCH_THRESHOLD = 0.82
CATEGORY2_MATCH_THRESHOLD = 0.72


class OpenBGSynchronizer:
    def __init__(self):
        self.writer = Neo4jWriter()
        self.base_attr_name_map = {}
        self.sale_attr_name_map = {}
        self.trademark_map = {}
        self.category2_candidates = []
        self.category3_candidates = []

    def close(self):
        self.writer.close()

    def run(self):
        rows = load_openbg_rows(OPENBG_JSONL_FILE)
        self._load_existing_maps()
        self.writer.create_constraints(
            GRAPH_NODE_LABELS + ["BaseAttrName", "BaseAttrValue", "SaleAttrName", "SaleAttrValue"]
        )

        payload = self._build_graph_payload(rows)
        self._clear_existing_relations(
            payload["spu_ids"],
            payload["sku_ids"],
            payload["category3_ids"],
        )
        self._write_payload(payload)

    def _load_existing_maps(self):
        self.base_attr_name_map = self._load_name_to_id("BaseAttrName")
        self.sale_attr_name_map = self._load_name_to_id("SaleAttrName")
        self.trademark_map = self._load_name_to_id("Trademark")
        self.category2_candidates = [
            Category2Candidate(id=item["id"], name=item["name"])
            for item in self.writer.query("MATCH (n:Category2) RETURN n.id AS id, n.name AS name")
            if item.get("id") is not None and item.get("name")
        ]
        self.category3_candidates = [
            Category3Candidate(
                id=item["id"],
                name=item["name"],
                parent_id=item["parent_id"],
                parent_name=item["parent_name"],
                source=item.get("source", ""),
            )
            for item in self.writer.query(
                """
                MATCH (c3:Category3)-[:Belong]->(c2:Category2)
                RETURN c3.id AS id,
                       c3.name AS name,
                       c2.id AS parent_id,
                       c2.name AS parent_name,
                       coalesce(c3.source, '') AS source
                """
            )
            if item.get("id") is not None and item.get("name") and item.get("parent_id") is not None
        ]

    def _load_name_to_id(self, label: str) -> dict[str, object]:
        rows = self.writer.query(f"MATCH (n:{label}) RETURN n.id AS id, n.name AS name")
        return {
            normalize_text(str(row["name"])): row["id"]
            for row in rows
            if row.get("id") is not None and row.get("name")
        }

    def _build_graph_payload(self, rows: list[dict]) -> dict:
        nodes_by_label = defaultdict(list)
        relations = []
        spu_ids = []
        sku_ids = []
        category3_ids = set()
        created_category3_ids = set()
        created_base_attr_names = set()
        created_base_attr_values = set()
        created_sale_attr_names = set()
        created_sale_attr_values = set()
        created_trademarks = set()

        for row in rows:
            item_id = str(row["item_id"])
            title = str(row.get("title", "")).strip()
            spu_id = f"openbg_spu::{item_id}"
            sku_id = f"openbg_sku::{item_id}::default"
            spu_ids.append(spu_id)
            sku_ids.append(sku_id)

            item_pairs = parse_pvs(row.get("item_pvs"))
            sku_pairs = parse_pvs(row.get("sku_pvs"))
            brand_name = extract_brand(item_pairs) or extract_brand(sku_pairs)

            category3_id, new_category3, category3_parent_relation = self._resolve_category(row)
            if new_category3 and new_category3["id"] not in created_category3_ids:
                nodes_by_label["Category3"].append(new_category3)
                created_category3_ids.add(new_category3["id"])
            if category3_parent_relation:
                relations.append(
                    {
                        "type": "Belong",
                        "start_label": "Category3",
                        "end_label": "Category2",
                        "batch": [category3_parent_relation],
                    }
                )

            nodes_by_label["SPU"].append(
                {
                    "id": spu_id,
                    "name": title,
                    "source": "openbg",
                    "item_id": item_id,
                    "title": title,
                    "industry_name": row.get("industry_name", ""),
                    "cate_id": row.get("cate_id", ""),
                    "cate_name": row.get("cate_name", ""),
                    "cate_id_path": row.get("cate_id_path", ""),
                    "cate_name_path": row.get("cate_name_path", ""),
                    "item_pvs_raw": row.get("item_pvs", ""),
                    "sku_pvs_raw": row.get("sku_pvs", ""),
                    "image_name": row.get("item_image_name", ""),
                    "description": "",
                }
            )
            nodes_by_label["SKU"].append(
                {
                    "id": sku_id,
                    "name": title,
                    "source": "openbg",
                    "item_id": item_id,
                    "item_pvs_raw": row.get("item_pvs", ""),
                    "sku_pvs_raw": row.get("sku_pvs", ""),
                }
            )

            relations.append(
                {
                    "type": "Belong",
                    "start_label": "SKU",
                    "end_label": "SPU",
                    "batch": [{"start_id": sku_id, "end_id": spu_id}],
                }
            )

            if category3_id:
                category3_ids.add(category3_id)
                relations.append(
                    {
                        "type": "Belong",
                        "start_label": "SPU",
                        "end_label": "Category3",
                        "batch": [{"start_id": spu_id, "end_id": category3_id}],
                    }
                )

            if brand_name:
                trademark_id = self.trademark_map.get(normalize_text(brand_name))
                if trademark_id is None:
                    trademark_id = stable_id("openbg_tm", brand_name)
                    self.trademark_map[normalize_text(brand_name)] = trademark_id
                    if trademark_id not in created_trademarks:
                        nodes_by_label["Trademark"].append(
                            {
                                "id": trademark_id,
                                "name": brand_name,
                                "source": "openbg",
                            }
                        )
                        created_trademarks.add(trademark_id)

                relations.append(
                    {
                        "type": "Belong",
                        "start_label": "SPU",
                        "end_label": "Trademark",
                        "batch": [{"start_id": spu_id, "end_id": trademark_id}],
                    }
                )

            sale_attr_name_ids = set()
            for key, value in item_pairs:
                (
                    base_name_id,
                    base_value_id,
                    base_name_node,
                    base_value_node,
                    is_new_base_name,
                ) = self._build_base_attr(key, value)
                if is_new_base_name and base_name_id not in created_base_attr_names:
                    nodes_by_label["BaseAttrName"].append(base_name_node)
                    created_base_attr_names.add(base_name_id)
                if base_value_id not in created_base_attr_values:
                    nodes_by_label["BaseAttrValue"].append(base_value_node)
                    created_base_attr_values.add(base_value_id)
                relations.append(
                    {
                        "type": "Have",
                        "start_label": "BaseAttrName",
                        "end_label": "BaseAttrValue",
                        "batch": [{"start_id": base_name_id, "end_id": base_value_id}],
                    }
                )
                relations.append(
                    {
                        "type": "Have",
                        "start_label": "SPU",
                        "end_label": "BaseAttrValue",
                        "batch": [{"start_id": spu_id, "end_id": base_value_id}],
                    }
                )

            for key, value in sku_pairs:
                if key in SKU_SALE_ATTR_WHITELIST:
                    (
                        sale_name_id,
                        sale_value_id,
                        sale_name_node,
                        sale_value_node,
                        is_new_sale_name,
                    ) = self._build_sale_attr(key, value)
                    if is_new_sale_name and sale_name_id not in created_sale_attr_names:
                        nodes_by_label["SaleAttrName"].append(sale_name_node)
                        created_sale_attr_names.add(sale_name_id)
                    if sale_value_id not in created_sale_attr_values:
                        nodes_by_label["SaleAttrValue"].append(sale_value_node)
                        created_sale_attr_values.add(sale_value_id)
                    relations.append(
                        {
                            "type": "Have",
                            "start_label": "SaleAttrName",
                            "end_label": "SaleAttrValue",
                            "batch": [{"start_id": sale_name_id, "end_id": sale_value_id}],
                        }
                    )
                    relations.append(
                        {
                            "type": "Have",
                            "start_label": "SKU",
                            "end_label": "SaleAttrValue",
                            "batch": [{"start_id": sku_id, "end_id": sale_value_id}],
                        }
                    )
                    sale_attr_name_ids.add(sale_name_id)
                    continue

                (
                    base_name_id,
                    base_value_id,
                    base_name_node,
                    base_value_node,
                    is_new_base_name,
                ) = self._build_base_attr(key, value)
                if is_new_base_name and base_name_id not in created_base_attr_names:
                    nodes_by_label["BaseAttrName"].append(base_name_node)
                    created_base_attr_names.add(base_name_id)
                if base_value_id not in created_base_attr_values:
                    nodes_by_label["BaseAttrValue"].append(base_value_node)
                    created_base_attr_values.add(base_value_id)
                relations.append(
                    {
                        "type": "Have",
                        "start_label": "BaseAttrName",
                        "end_label": "BaseAttrValue",
                        "batch": [{"start_id": base_name_id, "end_id": base_value_id}],
                    }
                )
                relations.append(
                    {
                        "type": "Have",
                        "start_label": "SKU",
                        "end_label": "BaseAttrValue",
                        "batch": [{"start_id": sku_id, "end_id": base_value_id}],
                    }
                )

            for sale_name_id in sale_attr_name_ids:
                relations.append(
                    {
                        "type": "Have",
                        "start_label": "SPU",
                        "end_label": "SaleAttrName",
                        "batch": [{"start_id": spu_id, "end_id": sale_name_id}],
                    }
                )

        return {
            "nodes_by_label": dict(nodes_by_label),
            "relations": relations,
            "spu_ids": spu_ids,
            "sku_ids": sku_ids,
            "category3_ids": list(category3_ids),
        }

    def _resolve_category(self, row: dict) -> tuple[str | None, dict | None, dict | None]:
        cate_name = str(row.get("cate_name", "")).strip()
        cate_name_path = str(row.get("cate_name_path", "")).strip()
        industry_name = str(row.get("industry_name", "")).strip()
        cate_id = str(row.get("cate_id", "")).strip()

        candidate3, score3 = choose_best_category3(cate_name, cate_name_path, self.category3_candidates)
        if candidate3 and score3 >= CATEGORY3_MATCH_THRESHOLD:
            parent_relation = None
            if candidate3.source == "openbg":
                parent_relation = {
                    "start_id": candidate3.id,
                    "end_id": candidate3.parent_id,
                }
            return candidate3.id, None, parent_relation

        candidate2, score2 = choose_best_category2(
            cate_name,
            cate_name_path,
            industry_name,
            self.category2_candidates,
        )
        if not candidate2 or score2 < CATEGORY2_MATCH_THRESHOLD:
            return None, None, None

        category3_key = f"{cate_id or 'unknown'}::{normalize_text(cate_name) or cate_name}"
        category3_id = stable_id("openbg_category3", category3_key)
        return (
            category3_id,
            {
                "id": category3_id,
                "name": cate_name,
                "source": "openbg",
                "source_cate_id": cate_id,
                "source_cate_name": cate_name,
                "source_cate_name_path": cate_name_path,
                "parent_id": candidate2.id,
            },
            {
                "start_id": category3_id,
                "end_id": candidate2.id,
            },
        )

    def _build_base_attr(self, key: str, value: str) -> tuple[object, str, dict, dict, bool]:
        normalized_key = normalize_text(key)
        is_new = normalized_key not in self.base_attr_name_map
        base_name_id = self.base_attr_name_map.get(normalized_key)
        if base_name_id is None:
            base_name_id = stable_id("openbg_base_attr_name", key)
            self.base_attr_name_map[normalized_key] = base_name_id

        base_value_id = stable_id("openbg_base_attr_value", f"{base_name_id}::{value}")
        return (
            base_name_id,
            base_value_id,
            {
                "id": base_name_id,
                "name": key,
                "source": "openbg",
            },
            {
                "id": base_value_id,
                "name": value,
                "source": "openbg",
                "attr_name": key,
            },
            is_new,
        )

    def _build_sale_attr(self, key: str, value: str) -> tuple[object, str, dict, dict, bool]:
        normalized_key = normalize_text(key)
        is_new = normalized_key not in self.sale_attr_name_map
        sale_name_id = self.sale_attr_name_map.get(normalized_key)
        if sale_name_id is None:
            sale_name_id = stable_id("openbg_sale_attr_name", key)
            self.sale_attr_name_map[normalized_key] = sale_name_id

        sale_value_id = stable_id("openbg_sale_attr_value", f"{sale_name_id}::{value}")
        return (
            sale_name_id,
            sale_value_id,
            {
                "id": sale_name_id,
                "name": key,
                "source": "openbg",
            },
            {
                "id": sale_value_id,
                "name": value,
                "source": "openbg",
                "sale_attr_name": key,
            },
            is_new,
        )

    def _clear_existing_relations(self, spu_ids: list[str], sku_ids: list[str], category3_ids: list[str]):
        self.writer.run_query(
            """
            MATCH (c3:Category3)-[r:Belong]->(:Category2)
            WHERE c3.id IN $category3_ids
            DELETE r
            """,
            category3_ids=category3_ids,
        )
        self.writer.run_query(
            """
            MATCH (s:SPU)-[r:Belong]->(t)
            WHERE s.id IN $spu_ids
              AND any(label IN labels(t) WHERE label IN ['Category3', 'Trademark'])
            DELETE r
            """,
            spu_ids=spu_ids,
        )
        self.writer.run_query(
            """
            MATCH (s:SPU)-[r:Have]->(t)
            WHERE s.id IN $spu_ids
              AND any(label IN labels(t) WHERE label IN ['BaseAttrValue', 'SaleAttrName'])
            DELETE r
            """,
            spu_ids=spu_ids,
        )
        self.writer.run_query(
            """
            MATCH (s:SKU)-[r:Have]->(t)
            WHERE s.id IN $sku_ids
              AND any(label IN labels(t) WHERE label IN ['BaseAttrValue', 'SaleAttrValue'])
            DELETE r
            """,
            sku_ids=sku_ids,
        )

    def _write_payload(self, payload: dict):
        for label, nodes in payload["nodes_by_label"].items():
            self.writer.write_nodes(label, nodes)

        relation_batches = defaultdict(list)
        for relation in payload["relations"]:
            key = (relation["type"], relation["start_label"], relation["end_label"])
            relation_batches[key].extend(relation["batch"])

        for (relation_type, start_label, end_label), batch in relation_batches.items():
            self.writer.write_relations(
                relation_type,
                start_label,
                end_label,
                batch,
            )


if __name__ == "__main__":
    synchronizer = OpenBGSynchronizer()
    try:
        synchronizer.run()
    finally:
        synchronizer.close()
