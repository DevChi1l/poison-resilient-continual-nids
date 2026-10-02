"""NumPy-only bounded-confusion and external-class mapping tests."""

import unittest

import numpy as np

from src.evaluation.classification import classification_metrics
from src.evaluation.streaming import ConfusionAccumulator


class StreamingEvaluationTests(unittest.TestCase):
    def test_chunked_counts_match_existing_evaluator_and_balanced_accuracy(self):
        truth = np.array([10, 10, 0, 0, 5, 5, 5], dtype=np.int64)
        predicted = np.array([10, 0, 0, 5, 5, 10, 5], dtype=np.int64)
        names = {10: "Benign", 0: "Attack A", 5: "Attack B"}
        accumulator = ConfusionAccumulator((10, 0, 5))
        truth_before = truth.copy()
        for start in (0, 3, 6):
            end = min(start + 3, len(truth))
            accumulator.update(truth[start:end], predicted[start:end])
        result = accumulator.metrics(class_names=names, benign_class_id=10)
        existing = classification_metrics(truth, predicted, class_ids=(10, 0, 5),
                                          class_names=names, benign_class_id=10)
        for key in ("accuracy", "macro_f1", "per_class", "confusion_matrix",
                    "benign_false_positive"):
            self.assertEqual(result[key], existing[key])
        self.assertAlmostEqual(result["balanced_accuracy"],
                               np.mean([0.5, 0.5, 2 / 3]))
        self.assertTrue(np.array_equal(truth, truth_before))

    def test_invalid_ids_and_empty_metrics(self):
        with self.assertRaises(ValueError):
            ConfusionAccumulator((0, 0))
        accumulator = ConfusionAccumulator((0, 2))
        with self.assertRaises(ValueError):
            accumulator.metrics()
        with self.assertRaises(ValueError):
            accumulator.update(np.array([0]), np.array([3]))
        with self.assertRaises(ValueError):
            accumulator.update(np.array([0, 2]), np.array([0]))
        accumulator.update(np.array([0, 0]), np.array([2, 2]))
        result = accumulator.metrics(benign_class_id=0)
        self.assertEqual(result["balanced_accuracy"], 0.0)
        self.assertEqual(result["benign_false_positive"]["rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
