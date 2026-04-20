import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"

if load_dotenv is not None:
    load_dotenv(ROOT_DIR / ".env")

DATA_DIR = ROOT_DIR / "data"
NER_DIR = "ner"
RAW_DATA_DIR = DATA_DIR / NER_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / NER_DIR / "processed"
NORMALIZATION_CONFIG_FILE = ROOT_DIR / "src" / "configuration" / "entity_normalization.json"

LOG_DIR = ROOT_DIR / "logs"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
BEST_MODEL_DIR = CHECKPOINT_DIR / NER_DIR / "best_model"

WEB_STATIC_DIR = ROOT_DIR / "src" / "web" / "static"

RAW_DATA_FILE = RAW_DATA_DIR / "data.json"
MODEL_NAME = "google-bert/bert-base-chinese"

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 3e-5
SAVE_STEPS = 100
EARLY_STOPPING_PATIENCE = 8
MAX_LENGTH = 128
SEED = 42

ENTITY_TYPES = ["ATTR", "PEOPLE", "SPEC"]
LABELS = [
    "O",
    *[f"{prefix}-{entity_type}" for entity_type in ENTITY_TYPES for prefix in ("B", "I")],
]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for index, label in enumerate(LABELS)}

TEXT_ENTITY_NODE_LABELS = {
    "ATTR": "AttributeTag",
    "PEOPLE": "PeopleTag",
    "SPEC": "SpecTag",
}

MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DATABASE", "gmall"),
    "charset": "utf8mb4",
}

NEO4J_CONFIG = {
    "uri": os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
    "auth": (
        os.getenv("NEO4J_USER", "neo4j"),
        os.getenv("NEO4J_PASSWORD"),
    ),
}

GRAPH_NODE_LABELS = [
    "Category1",
    "Category2",
    "Category3",
    "BaseAttrName",
    "BaseAttrValue",
    "SPU",
    "SKU",
    "Trademark",
    "SaleAttrName",
    "SaleAttrValue",
    *TEXT_ENTITY_NODE_LABELS.values(),
]

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "BAAI/bge-base-zh-v1.5",
)

ENTITY_INDEX_CONFIG = {
    "Trademark": {
        "fulltext_index": "trademark_fulltext_index",
        "vector_index": "trademark_vector_index",
    },
    "SPU": {
        "fulltext_index": "spu_fulltext_index",
        "vector_index": "spu_vector_index",
    },
    "SKU": {
        "fulltext_index": "sku_fulltext_index",
        "vector_index": "sku_vector_index",
    },
    "Category1": {
        "fulltext_index": "category1_fulltext_index",
        "vector_index": "category1_vector_index",
    },
    "Category2": {
        "fulltext_index": "category2_fulltext_index",
        "vector_index": "category2_vector_index",
    },
    "Category3": {
        "fulltext_index": "category3_fulltext_index",
        "vector_index": "category3_vector_index",
    },
    "AttributeTag": {
        "fulltext_index": "attribute_tag_fulltext_index",
        "vector_index": "attribute_tag_vector_index",
    },
    "PeopleTag": {
        "fulltext_index": "people_tag_fulltext_index",
        "vector_index": "people_tag_vector_index",
    },
    "SpecTag": {
        "fulltext_index": "spec_tag_fulltext_index",
        "vector_index": "spec_tag_vector_index",
    },
}


def ensure_project_dirs() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
