from collections import defaultdict

from datasync.utils import Neo4jWriter
from dialogue.types import RecommendationItem


USE_CASE_KEYWORDS = {
    "拍照": ("拍照", "影像", "自拍", "镜头", "徕卡", "摄影"),
    "游戏": ("游戏", "电竞", "性能", "旗舰处理器", "高刷", "顺滑"),
    "续航": ("续航", "大电量", "快充", "充电", "长续航"),
    "性价比": ("性价比", "全能", "实惠", "高配", "划算"),
}
PHONE_CATEGORY3_ID = 61
STORAGE_ATTR_NAME = "机身内存"


class PhoneGuideRetriever:
    def __init__(self):
        self.writer = Neo4jWriter()
        self._brand_vocabulary: list[str] | None = None
        self._storage_vocabulary: list[str] | None = None

    def close(self) -> None:
        self.writer.close()

    def load_brand_vocabulary(self) -> list[str]:
        if self._brand_vocabulary is None:
            rows = self.writer.query(
                """
                MATCH (:Category3 {id: $category_id})<-[:Belong]-(spu:SPU)-[:Belong]->(tm:Trademark)
                RETURN DISTINCT tm.name AS brand
                ORDER BY brand
                """,
                category_id=PHONE_CATEGORY3_ID,
            )
            self._brand_vocabulary = [row["brand"] for row in rows if row.get("brand")]
        return list(self._brand_vocabulary)

    def load_storage_vocabulary(self) -> list[str]:
        if self._storage_vocabulary is None:
            rows = self.writer.query(
                """
                MATCH (:Category3 {id: $category_id})<-[:Belong]-(spu:SPU)<-[:Belong]-(sku:SKU)
                MATCH (sku)-[:Have]->(value:BaseAttrValue)<-[:Have]-(name:BaseAttrName)
                WHERE name.name = $attr_name
                RETURN DISTINCT value.name AS storage
                ORDER BY storage
                """,
                category_id=PHONE_CATEGORY3_ID,
                attr_name=STORAGE_ATTR_NAME,
            )
            self._storage_vocabulary = [row["storage"] for row in rows if row.get("storage")]
        return list(self._storage_vocabulary)

    def search(self, slots: dict, limit: int = 3) -> list[RecommendationItem]:
        rows = self._fetch_candidate_rows(slots)
        if not rows:
            return []

        grouped = defaultdict(list)
        for row in rows:
            grouped[row["spu_id"]].append(row)

        candidates = []
        for spu_rows in grouped.values():
            best_row = self._choose_representative_row(spu_rows, slots)
            item = self._build_item(best_row, spu_rows, slots)
            score = self._score_item(item, slots)
            candidates.append((score, item))

        candidates.sort(key=lambda pair: (-pair[0], pair[1].price, pair[1].spu_id))
        return [item for _, item in candidates[:limit]]

    def compare(self, spu_ids: list[int], use_case: str | None) -> str:
        if len(spu_ids) < 2:
            return "当前候选不足 2 款，先让我给你推荐至少两款手机。"

        rows = self.writer.query(
            """
            MATCH (spu:SPU)-[:Belong]->(:Category3 {id: $category_id})
            MATCH (spu)-[:Belong]->(tm:Trademark)
            MATCH (sku:SKU)-[:Belong]->(spu)
            WHERE spu.id IN $spu_ids
              AND coalesce(sku.is_sale, 0) = 1
            OPTIONAL MATCH (sku)-[:Have]->(storageValue:BaseAttrValue)<-[:Have]-(storageName:BaseAttrName)
            WHERE storageName.name = $attr_name
            WITH spu, tm, sku, collect(DISTINCT storageValue.name) AS storage_values
            ORDER BY sku.price ASC
            RETURN spu.id AS spu_id,
                   spu.name AS spu_name,
                   coalesce(spu.description, '') AS description,
                   tm.name AS brand,
                   collect({
                       sku_id: sku.id,
                       sku_name: sku.name,
                       price: toFloat(sku.price),
                       storage_values: storage_values
                   }) AS sku_rows
            """,
            category_id=PHONE_CATEGORY3_ID,
            spu_ids=spu_ids[:2],
            attr_name=STORAGE_ATTR_NAME,
        )
        rows_by_spu = {row["spu_id"]: row for row in rows}
        first = rows_by_spu.get(spu_ids[0])
        second = rows_by_spu.get(spu_ids[1])
        if first is None or second is None:
            return "当前候选里缺少可比较的在售 SKU，换一组候选再试一次。"

        return self._format_compare_text(first, second, use_case)

    def get_min_price(self, brand: str | None = None) -> int | None:
        rows = self.writer.query(
            """
            MATCH (:Category3 {id: $category_id})<-[:Belong]-(spu:SPU)-[:Belong]->(tm:Trademark)
            MATCH (sku:SKU)-[:Belong]->(spu)
            WHERE coalesce(sku.is_sale, 0) = 1
              AND ($brand IS NULL OR tm.name = $brand)
            RETURN min(toInteger(sku.price)) AS min_price
            """,
            category_id=PHONE_CATEGORY3_ID,
            brand=brand,
        )
        if not rows:
            return None
        min_price = rows[0].get("min_price")
        return int(min_price) if min_price is not None else None

    def _fetch_candidate_rows(self, slots: dict) -> list[dict]:
        return self.writer.query(
            """
            MATCH (:Category3 {id: $category_id})<-[:Belong]-(spu:SPU)-[:Belong]->(tm:Trademark)
            MATCH (sku:SKU)-[:Belong]->(spu)
            WHERE coalesce(sku.is_sale, 0) = 1
              AND ($budget_max IS NULL OR toFloat(sku.price) <= $budget_max)
              AND ($brand IS NULL OR tm.name = $brand)
            OPTIONAL MATCH (sku)-[:Have]->(storageValue:BaseAttrValue)<-[:Have]-(storageName:BaseAttrName)
            WHERE storageName.name = $attr_name
            WITH sku, spu, tm, collect(DISTINCT storageValue.name) AS storage_values
            WHERE $storage IS NULL OR $storage IN storage_values
            RETURN sku.id AS sku_id,
                   sku.name AS sku_name,
                   coalesce(sku.price, 0) AS price,
                   coalesce(sku.sku_desc, '') AS sku_desc,
                   coalesce(sku.default_img, '') AS default_img,
                   spu.id AS spu_id,
                   spu.name AS spu_name,
                   coalesce(spu.description, '') AS spu_description,
                   tm.name AS brand,
                   storage_values
            """,
            category_id=PHONE_CATEGORY3_ID,
            budget_max=slots.get("budget_max"),
            brand=slots.get("brand"),
            storage=slots.get("storage"),
            attr_name=STORAGE_ATTR_NAME,
        )

    @staticmethod
    def _choose_representative_row(rows: list[dict], slots: dict) -> dict:
        requested_storage = slots.get("storage")
        if requested_storage:
            storage_matches = [row for row in rows if requested_storage in row.get("storage_values", [])]
            if storage_matches:
                rows = storage_matches
        rows.sort(key=lambda row: (float(row["price"]), row["sku_id"]))
        return rows[0]

    def _build_item(self, row: dict, grouped_rows: list[dict], slots: dict) -> RecommendationItem:
        price = float(row["price"])
        reason_parts = [f"满足预算 {int(slots['budget_max'])} 元以内"] if slots.get("budget_max") else []
        if slots.get("brand"):
            reason_parts.append(f"品牌为 {slots['brand']}")
        if slots.get("storage"):
            reason_parts.append(f"机身存储包含 {slots['storage']}")
        use_case = slots.get("use_case")
        if use_case:
            reason_parts.append(self._format_use_case_reason(use_case, row))
        reason = "，".join(reason_parts) if reason_parts else "符合当前导购条件"
        source_text = " ".join(
            part
            for part in (
                row.get("sku_name", ""),
                row.get("sku_desc", ""),
                row.get("spu_name", ""),
                row.get("spu_description", ""),
            )
            if part
        )
        return RecommendationItem(
            sku_id=int(row["sku_id"]),
            spu_id=int(row["spu_id"]),
            sku_name=row["sku_name"],
            spu_name=row["spu_name"],
            brand=row["brand"],
            price=price,
            reason=reason,
            default_img=row.get("default_img", ""),
            storage_options=self._collect_storage_options(grouped_rows),
            source_text=source_text,
        )

    def _score_item(self, item: RecommendationItem, slots: dict) -> float:
        score = 0.0
        use_case = slots.get("use_case")
        if use_case:
            text = item.source_text.lower()
            for keyword in USE_CASE_KEYWORDS[use_case]:
                if keyword.lower() in text:
                    score += 3.0

        budget_max = slots.get("budget_max")
        if budget_max:
            budget_gap = abs(budget_max - item.price)
            score += max(0.0, 2.5 - budget_gap / max(budget_max, 1) * 2.5)
            if use_case == "性价比":
                score += max(0.0, 1.5 - item.price / max(budget_max, 1))

        if slots.get("storage") and slots["storage"] in item.storage_options:
            score += 1.0
        return score

    @staticmethod
    def _format_use_case_reason(use_case: str, row: dict) -> str:
        text = f"{row['sku_name']} {row['sku_desc']} {row['spu_description']}"
        keywords = [keyword for keyword in USE_CASE_KEYWORDS[use_case] if keyword in text]
        if keywords:
            return f"{use_case}相关描述更强"
        return f"更贴近 {use_case} 诉求"

    @staticmethod
    def _collect_storage_options(rows: list[dict]) -> list[str]:
        return sorted(
            {
                storage
                for row in rows
                for storage in row.get("storage_values", [])
                if storage
            }
        )

    def _format_compare_text(self, first: dict, second: dict, use_case: str | None) -> str:
        first_best = first["sku_rows"][0]
        second_best = second["sku_rows"][0]
        first_storage = self._merge_storage_options(first["sku_rows"])
        second_storage = self._merge_storage_options(second["sku_rows"])

        lines = [
            f"1. {first['spu_name']}：品牌 {first['brand']}，当前可选最低价 {int(first_best['price'])} 元，可选存储 {first_storage}。",
            f"2. {second['spu_name']}：品牌 {second['brand']}，当前可选最低价 {int(second_best['price'])} 元，可选存储 {second_storage}。",
        ]
        if use_case:
            lines.append(f"按你当前更看重的“{use_case}”诉求，建议优先看描述里更贴近该用途的一款。")
        return "\n".join(lines)

    @staticmethod
    def _merge_storage_options(sku_rows: list[dict]) -> str:
        options = sorted(
            {
                storage
                for row in sku_rows
                for storage in row.get("storage_values", [])
                if storage
            }
        )
        return " / ".join(options) if options else "未标注"
