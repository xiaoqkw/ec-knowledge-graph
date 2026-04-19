import json
import re
from pathlib import Path

from configuration.config import NORMALIZATION_CONFIG_FILE, TEXT_ENTITY_NODE_LABELS


SPEC_UNIT_PATTERN = re.compile(
    r"(?i)(\d+(?:\.\d+)?)\s*(ml|l|g|kg|mg|cm|mm|m|片|袋|盒|支|包|瓶|粒|枚|双)"
)
WHITESPACE_PATTERN = re.compile(r"\s+")


class EntityNormalizer:
    """Apply lightweight canonicalization to extracted typed entities."""

    def __init__(self, config_file: Path | None = None):
        self.config_file = Path(config_file or NORMALIZATION_CONFIG_FILE)
        self.aliases = self._load_aliases()

    def normalize_entities(self, entities: list[dict]) -> list[dict]:
        normalized_entities = []
        for entity in entities:
            normalized = self.normalize_entity(entity)
            if normalized:
                normalized_entities.append(normalized)
        return normalized_entities

    def normalize_entity(self, entity: dict) -> dict | None:
        entity_type = str(entity.get("entity_type", "")).strip().upper()
        raw_text = str(entity.get("text", "")).strip()
        if not raw_text or entity_type not in TEXT_ENTITY_NODE_LABELS:
            return None

        canonical_name = self._normalize_text(entity_type, raw_text)
        if not canonical_name:
            return None

        normalized_entity = dict(entity)
        normalized_entity["entity_type"] = entity_type
        normalized_entity["raw_text"] = raw_text
        normalized_entity["canonical_name"] = canonical_name
        normalized_entity["node_label"] = TEXT_ENTITY_NODE_LABELS[entity_type]
        return normalized_entity

    def _load_aliases(self) -> dict[str, dict[str, str]]:
        if not self.config_file.exists():
            return {}

        with self.config_file.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        aliases_by_type = {}
        for entity_type, config in payload.items():
            aliases = config.get("aliases", {})
            aliases_by_type[entity_type.upper()] = {
                self._normalize_whitespace(key): self._normalize_whitespace(value)
                for key, value in aliases.items()
            }
        return aliases_by_type

    def _normalize_text(self, entity_type: str, text: str) -> str:
        normalized = self._normalize_whitespace(text)
        normalized = self.aliases.get(entity_type, {}).get(normalized, normalized)

        if entity_type == "SPEC":
            normalized = self._normalize_spec(normalized)

        return normalized

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return WHITESPACE_PATTERN.sub("", text).strip()

    @staticmethod
    def _normalize_spec(text: str) -> str:
        text = SPEC_UNIT_PATTERN.sub(
            lambda match: f"{match.group(1)}{match.group(2).lower()}",
            text,
        )
        text = text.replace("×", "x").replace("X", "x")
        return text
