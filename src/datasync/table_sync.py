import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve()
SRC_DIR = CURRENT_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datasync.utils import MysqlReader, Neo4jWriter



class TableSynchronizer:
    """把电商结构化表数据从 MySQL 同步到 Neo4j。"""

    def __init__(self):
        self.reader = MysqlReader()
        self.writer = Neo4jWriter()

    def close(self):
        self.reader.close()
        self.writer.close()

    def sync_category1(self):
        sql = """
            SELECT id, name
            FROM base_category1
        """
        self.writer.write_nodes("Category1", self.reader.read(sql))

    def sync_category2(self):
        sql = """
            SELECT id, name
            FROM base_category2
        """
        self.writer.write_nodes("Category2", self.reader.read(sql))

    def sync_category3(self):
        sql = """
            SELECT id, name
            FROM base_category3
        """
        self.writer.write_nodes("Category3", self.reader.read(sql))

    def sync_category2_to_category1(self):
        sql = """
            SELECT id AS start_id,
                   category1_id AS end_id
            FROM base_category2
        """
        self.writer.write_relations(
            "Belong",
            "Category2",
            "Category1",
            self.reader.read(sql),
        )

    def sync_category3_to_category2(self):
        sql = """
            SELECT id AS start_id,
                   category2_id AS end_id
            FROM base_category3
        """
        self.writer.write_relations(
            "Belong",
            "Category3",
            "Category2",
            self.reader.read(sql),
        )

    def sync_base_attr_name(self):
        sql = """
            SELECT id,
                   attr_name AS name
            FROM base_attr_info
        """
        self.writer.write_nodes("BaseAttrName", self.reader.read(sql))

    def sync_base_attr_value(self):
        sql = """
            SELECT id,
                   value_name AS name
            FROM base_attr_value
        """
        self.writer.write_nodes("BaseAttrValue", self.reader.read(sql))

    def sync_base_attr_name_to_value(self):
        sql = """
            SELECT attr_id AS start_id,
                   id AS end_id
            FROM base_attr_value
        """
        self.writer.write_relations(
            "Have",
            "BaseAttrName",
            "BaseAttrValue",
            self.reader.read(sql),
        )

    def sync_category1_to_base_attr_name(self):
        sql = """
            SELECT category_id AS start_id,
                   id AS end_id
            FROM base_attr_info
            WHERE category_level = 1
        """
        self.writer.write_relations(
            "Have",
            "Category1",
            "BaseAttrName",
            self.reader.read(sql),
        )

    def sync_category2_to_base_attr_name(self):
        sql = """
            SELECT category_id AS start_id,
                   id AS end_id
            FROM base_attr_info
            WHERE category_level = 2
        """
        self.writer.write_relations(
            "Have",
            "Category2",
            "BaseAttrName",
            self.reader.read(sql),
        )

    def sync_category3_to_base_attr_name(self):
        sql = """
            SELECT category_id AS start_id,
                   id AS end_id
            FROM base_attr_info
            WHERE category_level = 3
        """
        self.writer.write_relations(
            "Have",
            "Category3",
            "BaseAttrName",
            self.reader.read(sql),
        )

    def sync_spu(self):
        sql = """
            SELECT id,
                   spu_name AS name
            FROM spu_info
        """
        self.writer.write_nodes("SPU", self.reader.read(sql))

    def sync_sku(self):
        sql = """
            SELECT id,
                   sku_name AS name
            FROM sku_info
        """
        self.writer.write_nodes("SKU", self.reader.read(sql))

    def sync_sku_to_spu(self):
        sql = """
            SELECT id AS start_id,
                   spu_id AS end_id
            FROM sku_info
        """
        self.writer.write_relations("Belong", "SKU", "SPU", self.reader.read(sql))

    def sync_spu_to_category3(self):
        sql = """
            SELECT id AS start_id,
                   category3_id AS end_id
            FROM spu_info
        """
        self.writer.write_relations(
            "Belong",
            "SPU",
            "Category3",
            self.reader.read(sql),
        )

    def sync_trademark(self):
        sql = """
            SELECT id,
                   tm_name AS name
            FROM base_trademark
        """
        self.writer.write_nodes("Trademark", self.reader.read(sql))

    def sync_spu_to_trademark(self):
        sql = """
            SELECT id AS start_id,
                   tm_id AS end_id
            FROM spu_info
        """
        self.writer.write_relations(
            "Belong",
            "SPU",
            "Trademark",
            self.reader.read(sql),
        )

    def sync_sale_attr_name(self):
        sql = """
            SELECT id,
                   sale_attr_name AS name
            FROM spu_sale_attr
        """
        self.writer.write_nodes("SaleAttrName", self.reader.read(sql))

    def sync_sale_attr_value(self):
        sql = """
            SELECT id,
                   sale_attr_value_name AS name
            FROM spu_sale_attr_value
        """
        self.writer.write_nodes("SaleAttrValue", self.reader.read(sql))

    def sync_sale_attr_name_to_value(self):
        sql = """
            SELECT a.id AS start_id,
                   v.id AS end_id
            FROM spu_sale_attr a
            JOIN spu_sale_attr_value v
              ON a.spu_id = v.spu_id
             AND a.base_sale_attr_id = v.base_sale_attr_id
        """
        self.writer.write_relations(
            "Have",
            "SaleAttrName",
            "SaleAttrValue",
            self.reader.read(sql),
        )

    def sync_spu_to_sale_attr_name(self):
        sql = """
            SELECT spu_id AS start_id,
                   id AS end_id
            FROM spu_sale_attr
        """
        self.writer.write_relations(
            "Have",
            "SPU",
            "SaleAttrName",
            self.reader.read(sql),
        )

    def sync_sku_to_sale_attr_value(self):
        sql = """
            SELECT sku_id AS start_id,
                   sale_attr_value_id AS end_id
            FROM sku_sale_attr_value
        """
        self.writer.write_relations(
            "Have",
            "SKU",
            "SaleAttrValue",
            self.reader.read(sql),
        )

    def sync_sku_to_base_attr_value(self):
        sql = """
            SELECT sku_id AS start_id,
                   value_id AS end_id
            FROM sku_attr_value
        """
        self.writer.write_relations(
            "Have",
            "SKU",
            "BaseAttrValue",
            self.reader.read(sql),
        )

    def run_all(self):
        """按照依赖顺序执行完整的结构化数据同步。"""
        self.writer.create_constraints()

        # 先同步分类节点，再建立分类之间、分类与平台属性之间的关系。
        self.sync_category1()
        self.sync_category2()
        self.sync_category3()
        self.sync_category2_to_category1()
        self.sync_category3_to_category2()

        # 平台属性依赖分类节点存在后再写入关系。
        self.sync_base_attr_name()
        self.sync_base_attr_value()
        self.sync_base_attr_name_to_value()
        self.sync_category1_to_base_attr_name()
        self.sync_category2_to_base_attr_name()
        self.sync_category3_to_base_attr_name()

        # 商品和品牌信息依赖分类树，因此放在后面同步。
        self.sync_spu()
        self.sync_sku()
        self.sync_sku_to_spu()
        self.sync_spu_to_category3()
        self.sync_trademark()
        self.sync_spu_to_trademark()

        # 销售属性依赖 SPU / SKU，因此放在最后同步。
        self.sync_sale_attr_name()
        self.sync_sale_attr_value()
        self.sync_sale_attr_name_to_value()
        self.sync_spu_to_sale_attr_name()
        self.sync_sku_to_sale_attr_value()
        self.sync_sku_to_base_attr_value()


if __name__ == "__main__":
    synchronizer = TableSynchronizer()
    try:
        synchronizer.run_all()
    finally:
        synchronizer.close()
