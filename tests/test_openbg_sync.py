import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datasync.openbg_sync import OpenBGSynchronizer


class StubWriter:
    def __init__(self):
        self.calls = []

    def run_query(self, cypher: str, **parameters):
        self.calls.append((cypher, parameters))


class OpenBGSynchronizerTestCase(unittest.TestCase):
    def test_clear_existing_relations_only_deletes_openbg_category3_parent_edges(self):
        synchronizer = object.__new__(OpenBGSynchronizer)
        synchronizer.writer = StubWriter()

        synchronizer._clear_existing_relations(
            spu_ids=["openbg_spu::1"],
            sku_ids=["openbg_sku::1::default"],
            category3_ids=["c3_existing", "c3_openbg"],
        )

        first_query, first_params = synchronizer.writer.calls[0]
        self.assertIn("coalesce(c3.source, '') = 'openbg'", first_query)
        self.assertEqual(first_params["category3_ids"], ["c3_existing", "c3_openbg"])


if __name__ == "__main__":
    unittest.main()
