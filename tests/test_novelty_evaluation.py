"""Held-out unknown metrics, validation calibration, and ID mapping."""

import unittest

import numpy as np

from src.evaluation import novelty_metrics
from src.novelty import calibrate_confidence_threshold, detect_unknown, summarize_predictions


class HeldOutNoveltyTests(unittest.TestCase):
    def test_known_calibration_and_strict_boundary(self):
        probabilities = np.array([[0.8, 0.2], [0.6, 0.4], [0.7, 0.3]])
        original = probabilities.copy()
        calibration = calibrate_confidence_threshold(probabilities)
        self.assertEqual(calibration.quantile, 0.05)
        self.assertEqual(calibration.validation_count, 3)
        self.assertAlmostEqual(calibration.threshold, 0.61)
        np.testing.assert_array_equal(probabilities, original)
        np.testing.assert_array_equal(detect_unknown(np.array([[0.6, 0.4], [0.7, 0.3]]), 0.6), [False, False])
        for quantile in (True, -0.1, 1.1, float("nan")):
            with self.subTest(quantile=quantile), self.assertRaises(ValueError):
                calibrate_confidence_threshold(probabilities, quantile=quantile)
        with self.assertRaises(ValueError):
            calibrate_confidence_threshold(np.empty((0, 2)))

    def test_external_probability_columns_and_unknown_metrics(self):
        result = summarize_predictions(np.array([[0.25, 0.75], [0.9, 0.1]]), [20, 10], 0.8)
        np.testing.assert_array_equal(result.predicted_class_ids, [10, 20])
        np.testing.assert_array_equal(result.is_unknown, [True, False])
        truth = np.array([False, False, True, True, True])
        predicted = np.array([False, True, True, False, True])
        metrics = novelty_metrics(truth, predicted, true_class_ids=np.array([0, 1, 5, 6, 5]), unknown_class_ids=[5, 6])
        self.assertEqual((metrics["true_unknown_flagged"], metrics["known_false_rejected"], metrics["unknown_missed"]), (2, 1, 1))
        self.assertAlmostEqual(metrics["unknown_precision"], 2 / 3)
        self.assertAlmostEqual(metrics["unknown_recall"], 2 / 3)
        self.assertAlmostEqual(metrics["unknown_f1"], 2 / 3)
        self.assertEqual(metrics["known_false_rejection_rate"], 0.5)
        self.assertEqual(metrics["per_unknown_class"]["5"]["recall"], 1.0)
        self.assertEqual(metrics["per_unknown_class"]["6"]["recall"], 0.0)

    def test_edge_cases_and_validation(self):
        no_unknown = novelty_metrics(np.array([False, False]), np.array([False, False]))
        self.assertIsNone(no_unknown["unknown_recall"])
        self.assertIsNone(no_unknown["unknown_f1"])
        self.assertEqual(no_unknown["known_false_rejection_rate"], 0.0)
        all_unknown = novelty_metrics(np.array([True, True]), np.array([False, False]))
        self.assertEqual(all_unknown["unknown_precision"], 0.0)
        self.assertEqual(all_unknown["unknown_recall"], 0.0)
        self.assertEqual(all_unknown["unknown_f1"], 0.0)
        self.assertIsNone(all_unknown["known_false_rejection_rate"])
        empty = novelty_metrics(np.array([], dtype=bool), np.array([], dtype=bool))
        self.assertEqual(empty["unknown_true"], 0)
        for kwargs in (
            {"true_class_ids": np.array([0, 5]), "unknown_class_ids": [6]},
            {"true_class_ids": np.array([0, 5])},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                novelty_metrics(np.array([False, True]), np.array([False, True]), **kwargs)
        with self.assertRaises(ValueError):
            novelty_metrics(np.array([0, 1]), np.array([False, True]))


if __name__ == "__main__":
    unittest.main()
