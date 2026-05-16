import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from web.memory import CachedAlignedEntity, FailureMemory, InMemoryQASessionStore


class WebMemoryTestCase(unittest.TestCase):
    def test_commit_turn_updates_history_trace_and_task_memory(self):
        store = InMemoryQASessionStore()
        session = store.get_or_create("session-1")
        store.commit_turn(
            session,
            user_message="u1",
            assistant_message="a1",
            trace_id="trace-1",
            entity_cache_updates=[
                CachedAlignedEntity(
                    label="Trademark",
                    raw_entity="Apple",
                    aligned_entity="Apple Inc.",
                    param_name="param_0",
                    matched=True,
                )
            ],
            recent_failure=FailureMemory(
                trace_id="trace-1",
                failure_stage="query",
                primary_failure_tag="query_empty",
                cypher_excerpt="MATCH ...",
            ),
        )
        self.assertEqual(session.last_trace_id, "trace-1")
        self.assertEqual(session.history[-1]["assistant"], "a1")
        self.assertIn("Trademark::apple", session.task_memory.entity_cache.keys())
        self.assertEqual(session.task_memory.recent_failures[-1].primary_failure_tag, "query_empty")


if __name__ == "__main__":
    unittest.main()
