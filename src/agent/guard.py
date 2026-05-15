from __future__ import annotations

import re

from agent.types import ValidationResult


class CypherGuard:
    DENIED_TOKENS = {
        "CREATE",
        "MERGE",
        "SET",
        "DELETE",
        "REMOVE",
        "DROP",
        "CALL DBMS",
        "LOAD CSV",
    }
    _LIMIT_PATTERN = re.compile(r"(?i)\bLIMIT\s+(?P<limit>\d+)\b")
    _PARAM_PATTERN = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*")
    _WHERE_LITERAL_PATTERN = re.compile(
        r"(?is)\bWHERE\b.*?(?:=|<>|<=|>=|<|>|IN|CONTAINS|STARTS WITH|ENDS WITH)\s*(['\"]).+?\1"
    )

    def __init__(self, graph_client=None, *, graph_client_factory=None, enable_explain: bool = True, max_rows: int = 100):
        self.graph_client = graph_client
        self.graph_client_factory = graph_client_factory
        self.enable_explain = enable_explain
        self.max_rows = max_rows

    def validate(self, cypher: str, params: dict | None = None) -> ValidationResult:
        params = dict(params or {})
        normalized = (cypher or "").strip()
        if not normalized:
            return ValidationResult(ok=False, stage="empty_query", failure_mode="query_empty", error="Cypher query is empty.")

        if ";" in normalized.rstrip(";"):
            return ValidationResult(
                ok=False,
                stage="multi_statement",
                failure_mode="unsafe_query_blocked",
                error="Multiple statements are not allowed.",
                normalized_cypher=normalized,
            )

        upper_cypher = normalized.upper()
        if any(token in upper_cypher for token in self.DENIED_TOKENS):
            return ValidationResult(
                ok=False,
                stage="denied_token",
                failure_mode="unsafe_query_blocked",
                error="Detected denied Cypher token.",
                normalized_cypher=normalized,
            )

        if self._WHERE_LITERAL_PATTERN.search(normalized):
            return ValidationResult(
                ok=False,
                stage="non_parameterized_where",
                failure_mode="unsafe_query_blocked",
                error="String literals are not allowed in WHERE filters.",
                normalized_cypher=normalized,
            )

        normalized = self._enforce_limit(normalized, self.max_rows)

        if not self.enable_explain:
            return ValidationResult(ok=True, normalized_cypher=normalized, explain_checked=False)

        explain_params = self._build_explain_params(normalized, params)
        try:
            self._get_graph_client().query(f"EXPLAIN {normalized}", **explain_params)
        except Exception as exc:
            return ValidationResult(
                ok=False,
                stage="syntax_or_schema_invalid",
                failure_mode="query_validation_failed",
                error=str(exc),
                normalized_cypher=normalized,
                explain_checked=True,
            )
        return ValidationResult(
            ok=True,
            normalized_cypher=normalized,
            explain_checked=True,
        )

    def _get_graph_client(self):
        if self.graph_client is None:
            if self.graph_client_factory is None:
                raise RuntimeError("Graph client is not configured for CypherGuard.")
            self.graph_client = self.graph_client_factory()
        return self.graph_client

    def _build_explain_params(self, cypher: str, params: dict) -> dict:
        filled = dict(params)
        for param_name in self._PARAM_PATTERN.findall(cypher):
            key = param_name[1:]
            if key not in filled:
                filled[key] = "test"
        return filled

    def _enforce_limit(self, cypher: str, max_rows: int) -> str:
        match = self._LIMIT_PATTERN.search(cypher)
        if match is None:
            return f"{cypher}\nLIMIT {max_rows}"

        current = int(match.group("limit"))
        if current <= max_rows:
            return cypher
        return (
            cypher[: match.start("limit")]
            + str(max_rows)
            + cypher[match.end("limit") :]
        )
