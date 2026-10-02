"""Synthetic checks for deterministic training-label corruption."""

import unittest

import numpy as np

from src.poisoning import apply_label_flip


class LabelFlipTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([0, 1, 2, 1, 2, 0, 1, 2, 1, 2], dtype=np.int64)

    def test_random_is_reproducible_and_preserves_input(self):
        before = self.labels.copy()
        first = apply_label_flip(self.labels, 0.25, 42, allowed_classes=[0, 1, 2])
        second = apply_label_flip(self.labels, 0.25, 42, allowed_classes=[0, 1, 2])
        np.testing.assert_array_equal(self.labels, before)
        np.testing.assert_array_equal(first.changed_indices, second.changed_indices)
        np.testing.assert_array_equal(first.poisoned_labels, second.poisoned_labels)
        self.assertEqual(first.changed_count, 3)  # floor(2.5 + 0.5)
        self.assertEqual(first.eligible_count, 10)
        self.assertEqual(first.achieved_total_fraction, 0.3)
        self.assertTrue(np.all(first.poisoned_labels[first.changed_indices] != self.labels[first.changed_indices]))
        self.assertTrue(np.isin(first.poisoned_labels, [0, 1, 2]).all())
        self.assertFalse(np.shares_memory(first.poisoned_labels, self.labels))

    def test_targeted_uses_eligible_denominator_and_original_rows(self):
        rows = np.arange(100, 110, dtype=np.int64)
        before = rows.copy()
        first = apply_label_flip(self.labels, 0.5, 7, mode="targeted", source_class_ids=[1],
                                 target_class_id=0, allowed_class_ids=[0, 1, 2], original_row_indices=rows)
        second = apply_label_flip(self.labels, 0.5, 7, mode="targeted", source_class_ids=[1],
                                  target_class_id=0, allowed_class_ids=[0, 1, 2], original_row_indices=rows)
        self.assertEqual((first.eligible_count, first.changed_count), (4, 2))
        self.assertEqual(first.achieved_eligible_fraction, 0.5)
        self.assertEqual(first.achieved_total_fraction, 0.2)
        self.assertTrue(np.all(self.labels[first.changed_indices] == 1))
        self.assertTrue(np.all(first.poisoned_labels[first.changed_indices] == 0))
        np.testing.assert_array_equal(first.changed_original_row_indices, rows[first.changed_indices])
        np.testing.assert_array_equal(first.changed_indices, second.changed_indices)
        np.testing.assert_array_equal(self.labels, [0, 1, 2, 1, 2, 0, 1, 2, 1, 2])
        np.testing.assert_array_equal(rows, before)

    def test_zero_and_full_rates(self):
        zero = apply_label_flip(self.labels, 0, 42)
        self.assertEqual(zero.changed_count, 0)
        np.testing.assert_array_equal(zero.poisoned_labels, self.labels)
        self.assertFalse(np.shares_memory(zero.poisoned_labels, self.labels))
        full = apply_label_flip(self.labels, 1, 42, allowed_class_ids=[0, 1, 2])
        self.assertEqual(full.changed_count, len(self.labels))
        self.assertTrue(np.all(full.poisoned_labels != self.labels))
        targeted = apply_label_flip(self.labels, 1, 42, mode="targeted", source_class_ids=[1, 2],
                                    target_class_id=0, allowed_class_ids=[0, 1, 2])
        self.assertEqual(targeted.changed_count, 8)
        self.assertTrue(np.all(targeted.poisoned_labels == 0))

    def test_invalid_inputs(self):
        bad = [
            (self.labels.astype(float), 0.1, 1, {}),
            (self.labels.reshape(2, 5), 0.1, 1, {}),
            (self.labels, -0.1, 1, {}),
            (self.labels, 1.1, 1, {}),
            (self.labels, float("nan"), 1, {}),
            (self.labels, 0.1, True, {}),
            (self.labels, 0.1, 1, {"allowed_class_ids": [0, 1]}),
            (self.labels, 0.1, 1, {"allowed_class_ids": [0, 1, 1, 2]}),
            (self.labels, 0.1, 1, {"mode": "targeted", "source_class_ids": [1]}),
            (self.labels, 0.1, 1, {"mode": "targeted", "source_class_ids": [9], "target_class_id": 0}),
            (self.labels, 0.1, 1, {"mode": "targeted", "source_class_ids": [1], "target_class_id": 9}),
            (self.labels, 0.1, 1, {"original_row_indices": np.zeros(len(self.labels), dtype=int)}),
        ]
        for labels, rate, seed, kwargs in bad:
            with self.subTest(kwargs=kwargs, rate=rate):
                with self.assertRaises(ValueError):
                    apply_label_flip(labels, rate, seed, **kwargs)


if __name__ == "__main__":
    unittest.main()
