import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve()
SRC_DIR = CURRENT_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from configuration.config import GRAPH_NODE_LABELS
from datasync.utils import Neo4jWriter


def create_dialogue_indexes(writer: Neo4jWriter) -> None:
    writer.run_query(
        """
        CREATE RANGE INDEX sku_price_range_index IF NOT EXISTS
        FOR (n:SKU) ON (n.price)
        """
    )
    writer.run_query(
        """
        CREATE RANGE INDEX sku_is_sale_range_index IF NOT EXISTS
        FOR (n:SKU) ON (n.is_sale)
        """
    )


def create_schema() -> None:
    writer = Neo4jWriter()
    try:
        writer.create_constraints(GRAPH_NODE_LABELS)
        create_dialogue_indexes(writer)
        print("Neo4j constraints and dialogue indexes are ready.")
    finally:
        writer.close()


if __name__ == "__main__":
    create_schema()
