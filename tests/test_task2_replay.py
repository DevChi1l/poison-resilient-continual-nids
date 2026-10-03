"""Synthetic bounded Task 2 + replay sampling and scope checks."""

import unittest

import numpy as np

from src.training.task2_replay import Task2ReplayView


class FakeTask2:
    partition = "train"
    class_ids = (5, 6, 7)

    def __init__(self):
        self.features = np.arange(24, dtype=np.float32).reshape(12, 2)
        self.labels = np.array([0, 1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 0])
        self.splits = np.array([0] * 11 + [2])
        self.original_ids = np.arange(12, dtype=np.int64)
        self.indices = np.arange(5, 11, dtype=np.int64)

    def batch(self, rows):
        # Distinguishes transformed new rows from already-transformed replay.
        return self.features[rows] + 100, self.labels[rows].copy(), rows.copy()


class Task2ReplayTests(unittest.TestCase):
    def setUp(self):
        self.task2 = FakeTask2()
        self.replay_x = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        self.replay_y = np.array([0, 1, 0], dtype=np.int64)
        self.replay_ids = np.array([0, 1, 11], dtype=np.int64)

    def test_balanced_draws_no_double_transform_and_epoch_advancement(self):
        self.replay_ids[2] = 2
        original = self.replay_x.copy()
        view = Task2ReplayView(self.task2, self.replay_x, self.replay_y,
                               self.replay_ids, draws_per_epoch=96)
        self.assertEqual(view.present_class_ids, (0, 1, 5, 6, 7))
        first = list(view.iter_epoch(batch_size=13, seed=42, epoch=0))
        exposure = view.last_epoch_exposure
        self.assertEqual(sum(len(batch[1]) for batch in first), 96)
        self.assertEqual(sum(exposure["sampled_counts"].values()), 96)
        self.assertEqual(exposure["draws"], 96)
        self.assertLessEqual(exposure["unique_replay_rows_drawn"], 3)
        for X, y, ids in first:
            for row, label, raw_id in zip(X, y, ids):
                if label < 5:
                    self.assertTrue(np.array_equal(row, original[[0, 1, 2].index(raw_id)]))
                else:
                    self.assertTrue(np.array_equal(row, self.task2.features[raw_id] + 100))
        np.testing.assert_array_equal(self.replay_x, original)
        again = list(view.iter_epoch(batch_size=13, seed=42, epoch=0))
        self.assertTrue(all(np.array_equal(a[2], b[2]) for a, b in zip(first, again)))
        next_epoch = list(view.iter_epoch(batch_size=13, seed=42, epoch=1))
        self.assertFalse(all(np.array_equal(a[2], b[2]) for a, b in zip(first, next_epoch)))

    def test_filtered_empty_old_class_and_boundary_rejections(self):
        view = Task2ReplayView(self.task2, self.replay_x[:0], self.replay_y[:0],
                               self.replay_ids[:0], draws_per_epoch=17)
        self.assertEqual(view.present_class_ids, (5, 6, 7))
        batches = list(view.iter_epoch(batch_size=8, seed=7, epoch=0))
        self.assertEqual(sum(len(y) for _, y, _ in batches), 17)
        self.assertEqual(view.last_epoch_exposure["unique_replay_rows_drawn"], 0)
        with self.assertRaises(ValueError):
            Task2ReplayView(self.task2, self.replay_x, self.replay_y,
                            self.replay_ids, draws_per_epoch=10)  # ID 11 is test
        self.task2.partition = "validation"
        with self.assertRaises(ValueError):
            Task2ReplayView(self.task2, self.replay_x[:0], self.replay_y[:0],
                            self.replay_ids[:0], draws_per_epoch=10)


if __name__ == "__main__":
    unittest.main()
