import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve()
SRC_DIR = CURRENT_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from configuration.config import GRAPH_NODE_LABELS
from datasync.utils import Neo4jWriter


def create_schema() -> None:
    """创建同步脚本依赖的 Neo4j 唯一约束。"""
    writer = Neo4jWriter()
    try:
        writer.create_constraints(GRAPH_NODE_LABELS)
        print("Neo4j 唯一约束已创建，或本来就已存在。")
    finally:
        writer.close()


if __name__ == "__main__":
    create_schema()
