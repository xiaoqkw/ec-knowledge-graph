import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent.guard import CypherGuard


class StubGraphClient:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls = []

    def query(self, cypher, **params):
        self.calls.append((cypher, dict(params)))
        if self.error is not None:
            raise self.error
        return []


class CypherGuardTestCase(unittest.TestCase):
    def test_blocks_multi_statement(self):
        guard = CypherGuard(StubGraphClient(), enable_explain=False)
        result = guard.validate("MATCH (n) RETURN n; MATCH (m) RETURN m")
        self.assertFalse(result.ok)
        self.assertEqual(result.stage, "multi_statement")

    def test_blocks_denied_token(self):
        guard = CypherGuard(StubGraphClient(), enable_explain=False)
        result = guard.validate("MATCH (n) DELETE n")
        self.assertFalse(result.ok)
        self.assertEqual(result.stage, "denied_token")

    def test_enforces_limit(self):
        guard = CypherGuard(StubGraphClient(), enable_explain=False, max_rows=100)
        result = guard.validate("MATCH (n) RETURN n")
        self.assertTrue(result.ok)
        self.assertIn("LIMIT 100", result.normalized_cypher)

    def test_uses_explain_when_enabled(self):
        client = StubGraphClient()
        guard = CypherGuard(client, enable_explain=True)
        result = guard.validate("MATCH (n) RETURN n", {"param_0": "x"})
        self.assertTrue(result.ok)
        self.assertTrue(result.explain_checked)
        self.assertTrue(client.calls[0][0].startswith("EXPLAIN "))


if __name__ == "__main__":
    unittest.main()
