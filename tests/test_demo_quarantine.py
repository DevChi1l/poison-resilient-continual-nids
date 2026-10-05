from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from src.demo.quarantine import (
    QuarantineStore,
    ReplayBuffer,
    evaluate_targeted_gate,
)


class DemoQuarantineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        labels = np.repeat(np.arange(5, dtype=np.int64), 10)
        self.replay = ReplayBuffer(
            features=np.column_stack((labels, np.arange(50))).astype(np.float32),
            labels=labels,
            original_ids=np.arange(1000, 1050, dtype=np.int64),
        )
        probabilities = np.zeros((50, 5), dtype=np.float32)
        probabilities[np.arange(50), labels] = 1.0
        self.result = evaluate_targeted_gate(
            self.replay, probabilities, threshold=0.1, seed=42, rate=0.2
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_targeted_corruption_and_gate_are_deterministic(self) -> None:
        self.assertEqual(len(self.result.changed_indices), 8)
        self.assertEqual(self.result.audit["poison_rejected"], 8)
        self.assertEqual(self.result.audit["clean_false_rejected"], 0)
        self.assertEqual(self.result.audit["retained_total"], 42)
        repeated = evaluate_targeted_gate(
            self.replay,
            np.eye(5, dtype=np.float32)[self.replay.labels],
            threshold=0.1,
            seed=42,
            rate=0.2,
        )
        np.testing.assert_array_equal(self.result.changed_indices, repeated.changed_indices)
        np.testing.assert_array_equal(self.result.retained_mask, repeated.retained_mask)

    def test_queue_persists_rows_idempotently_and_audits_final_decision(self) -> None:
        database = Path(self.temporary.name) / "private/review.sqlite3"
        store = QuarantineStore(database)
        records = self.result.queue_records(self.replay, "teacher:sha")
        self.assertEqual(store.enqueue(records), 8)
        self.assertEqual(store.enqueue(records), 0)
        pending = store.items(status="pending")
        self.assertEqual(len(pending), 8)
        self.assertNotIn("changed", pending[0])
        item_id = pending[0]["id"]
        store.decide(item_id, "released", "reviewer-a", "verified label")
        self.assertEqual(store.items(status="released")[0]["status"], "released")
        history = store.history(item_id)
        self.assertEqual([row["decision"] for row in history], ["quarantined", "released"])
        with self.assertRaisesRegex(ValueError, "already finalized"):
            store.decide(item_id, "rejected", "reviewer-b", "second decision")


if __name__ == "__main__":
    unittest.main()
