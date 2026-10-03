"""Fixed-class forgetting and bounded confusion-combination checks."""

import unittest

import numpy as np

from src.evaluation.streaming import ConfusionAccumulator, summarize_task2_counts


class FullContinualMetricTests(unittest.TestCase):
    def test_new_class_predictions_remain_old_errors_and_forgetting_signed(self):
        before = ConfusionAccumulator(range(5))
        before.update(np.array([0, 1, 1]), np.array([0, 1, 0]))
        reference = before.metrics()
        old = np.zeros((8, 8), dtype=np.int64)
        old[0, 0] = 1
        old[1, 1] = 1
        old[1, 5] = 1  # old row misclassified as a genuinely new class
        new = np.zeros((8, 8), dtype=np.int64)
        new[5, 5] = 1
        new[6, 0] = 1
        names = {i: str(i) for i in range(8)}
        original_old = old.copy()
        result = summarize_task2_counts(reference, old, new, class_names=names)
        self.assertEqual(result["old_test"]["accuracy"], 2 / 3)
        self.assertEqual(result["combined_test"]["total_rows"], 5)
        self.assertEqual(result["old_test"]["confusion_matrix"][1][5], 1)
        self.assertEqual(result["forgetting"]["accuracy_forgetting"], 0.0)
        self.assertLess(result["forgetting"]["macro_f1_forgetting"], 0)
        self.assertEqual(result["old_attack_to_benign"]["support"], 2)
        np.testing.assert_array_equal(old, original_old)

    def test_mismatched_reference_or_partition_rejected(self):
        before = ConfusionAccumulator(range(5))
        before.update(np.array([0, 1]), np.array([0, 1]))
        reference = before.metrics()
        old = np.zeros((8, 8), dtype=np.int64)
        old[0, 0] = 1
        old[1, 1] = 1
        new = np.zeros((8, 8), dtype=np.int64)
        new[5, 5] = 1
        names = {i: str(i) for i in range(8)}
        summarize_task2_counts(reference, old, new, class_names=names)
        invalid = old.copy()
        invalid[0, 0] = 2
        with self.assertRaises(ValueError):
            summarize_task2_counts(reference, invalid, new, class_names=names)
        invalid = new.copy()
        invalid[0, 0] = 1
        with self.assertRaises(ValueError):
            summarize_task2_counts(reference, old, invalid, class_names=names)


if __name__ == "__main__":
    unittest.main()
