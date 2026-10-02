"""Synthetic checks for reusable known-class evaluation."""

import unittest

import numpy as np

from src.evaluation import classification_metrics


class EvaluationTests(unittest.TestCase):
    def test_metrics_and_attack_diagnostics(self):
        truth = np.array([0, 0, 4, 4, 9, 9])
        predicted = np.array([0, 4, 0, 4, 9, 0])
        result = classification_metrics(truth, predicted, class_ids=[0, 4, 9],
                                        benign_class_id=0, source_class_ids=[4, 9], target_class_id=0)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["confusion_matrix"], [[1, 1, 0], [1, 1, 0], [1, 0, 1]])
        self.assertEqual(result["benign_false_positive"]["rate"], 0.5)
        self.assertEqual(result["source_to_target_misclassification"]["count"], 2)
        self.assertEqual(result["source_to_target_misclassification"]["rate"], 0.5)
        self.assertEqual(result["per_class"]["9"]["recall"], 0.5)
        np.testing.assert_array_equal(truth, [0, 0, 4, 4, 9, 9])

    def test_rejects_invalid_inputs(self):
        for kwargs in ({"class_ids": [0, 0]}, {"class_ids": [0]},
                       {"source_class_ids": [1]}, {"benign_class_id": 4}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                classification_metrics(np.array([0, 1]), np.array([1, 0]), **kwargs)
        with self.assertRaises(ValueError):
            classification_metrics(np.array([], dtype=int), np.array([], dtype=int))


if __name__ == "__main__":
    unittest.main()
