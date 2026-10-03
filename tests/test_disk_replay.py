"""Synthetic training-only replay selection checks."""

from types import SimpleNamespace
import unittest

import numpy as np

from src.continual_learning.disk_replay import select_disk_replay_rows


class DiskReplayTests(unittest.TestCase):
    def setUp(self):
        self.view = SimpleNamespace(
            partition="train", class_ids=(0, 1, 2),
            indices=np.array([0, 2, 3, 5, 7, 8, 9, 11, 12], dtype=np.int64),
            labels=np.array([0, 7, 0, 1, 7, 1, 7, 2, 1, 2, 7, 2, 0]),
            splits=np.array([0, 2, 0, 0, 2, 0, 2, 0, 0, 0, 2, 0, 0]),
        )

    def test_balanced_deterministic_and_preserves_input(self):
        before_indices = self.view.indices.copy()
        before_labels = self.view.labels.copy()
        first = select_disk_replay_rows(self.view, class_ids=(0, 1, 2),
                                        per_class=2, seed=42)
        second = select_disk_replay_rows(self.view, class_ids=(0, 1, 2),
                                         per_class=2, seed=42)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(np.unique(first)), 6)
        self.assertEqual([np.count_nonzero(self.view.labels[first] == i)
                          for i in (0, 1, 2)], [2, 2, 2])
        self.assertTrue(np.all(self.view.splits[first] == 0))
        np.testing.assert_array_equal(self.view.indices, before_indices)
        np.testing.assert_array_equal(self.view.labels, before_labels)

    def test_rejects_future_or_validation_rows_and_unavailable_count(self):
        with self.assertRaises(ValueError):
            select_disk_replay_rows(self.view, class_ids=(0, 1, 2), per_class=4)
        with self.assertRaises(ValueError):
            select_disk_replay_rows(self.view, class_ids=(3,), per_class=1)
        self.view.partition = "validation"
        with self.assertRaises(ValueError):
            select_disk_replay_rows(self.view, class_ids=(0,), per_class=1)
        self.view.partition = "train"
        self.view.splits[0] = 2
        with self.assertRaises(ValueError):
            select_disk_replay_rows(self.view, class_ids=(0,), per_class=1)


if __name__ == "__main__":
    unittest.main()
