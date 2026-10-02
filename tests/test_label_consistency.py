"""Dataset-free tests for the frozen-teacher replay-label gate and its audit."""

import unittest

import numpy as np

from src.evaluation import replay_gate_metrics
from src.mitigation import (
    apply_label_consistency_gate,
    calibrate_label_consistency,
    label_inconsistency_scores,
)
from src.poisoning import apply_label_flip


class LabelConsistencyTests(unittest.TestCase):
    def test_external_class_mapping_and_preservation(self):
        probabilities = np.array([[0.1, 0.7, 0.2], [0.4, 0.1, 0.5]])
        labels = np.array([0, 5], dtype=np.int64)
        before = probabilities.copy()
        scores = label_inconsistency_scores(probabilities, labels, [10, 0, 5])
        np.testing.assert_allclose(scores, [0.3, 0.5])
        np.testing.assert_array_equal(probabilities, before)
        np.testing.assert_array_equal(labels, [0, 5])

    def test_predeclared_quantile_and_invalid_calibration(self):
        probabilities = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])
        labels = np.full(4, 7, dtype=np.int64)
        calibration = calibrate_label_consistency(probabilities, labels, [7, 9])
        self.assertEqual(calibration.quantile, 0.95)
        self.assertEqual(calibration.validation_count, 4)
        self.assertEqual(calibration.method, "linear")
        self.assertAlmostEqual(calibration.threshold, np.quantile([0.1, 0.2, 0.3, 0.4], 0.95))
        for quantile in (-0.1, 1.1, float("nan"), True):
            with self.subTest(quantile=quantile), self.assertRaises(ValueError):
                calibrate_label_consistency(probabilities, labels, [7, 9], quantile=quantile)
        with self.assertRaises(ValueError):
            calibrate_label_consistency(np.empty((0, 2)), np.empty(0, dtype=int), [7, 9])

    def test_gate_decision_equality_copies_and_empty_replay(self):
        features = np.arange(6, dtype=np.float32).reshape(3, 2)
        probabilities = np.array([[0.75, 0.125, 0.125], [0.75, 0.125, 0.125], [0.125, 0.125, 0.75]])
        labels = np.array([10, 0, 5], dtype=np.int64)
        original_features = features.copy()
        original_probabilities = probabilities.copy()
        original_labels = labels.copy()
        result = apply_label_consistency_gate(features, labels, probabilities, [10, 0, 5], 0.25)
        np.testing.assert_array_equal(result.retained_mask, [True, False, True])
        np.testing.assert_array_equal(result.retained_labels, [10, 5])
        np.testing.assert_array_equal(features, original_features)
        np.testing.assert_array_equal(probabilities, original_probabilities)
        np.testing.assert_array_equal(labels, original_labels)
        self.assertFalse(np.shares_memory(result.retained_features, features))
        copied_X, copied_y = result.as_fit_replay()
        copied_X[0, 0] = -99
        copied_y[0] = -99
        self.assertEqual(result.retained_features[0, 0], 0)
        self.assertEqual(result.retained_labels[0], 10)
        empty = apply_label_consistency_gate(features[:0], labels[:0], probabilities[:0], [10, 0, 5], 0.25)
        self.assertEqual(empty.retained_features.shape, (0, 2))
        self.assertIsNone(empty.as_fit_replay())
        all_rejected = apply_label_consistency_gate(features, labels, probabilities, [10, 0, 5], 0.0)
        self.assertIsNone(all_rejected.as_fit_replay())

    def test_rejects_bad_probabilities_and_mappings(self):
        good = np.array([[0.6, 0.4]])
        for probabilities, labels, ids in (
            (np.array([0.6, 0.4]), np.array([1]), [1, 2]),
            (np.array([[0.6, np.nan]]), np.array([1]), [1, 2]),
            (np.array([[0.6, 0.6]]), np.array([1]), [1, 2]),
            (good, np.array([3]), [1, 2]),
            (good, np.array([1]), [1, 1]),
            (good, np.array([1]), [1]),
        ):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                label_inconsistency_scores(probabilities, labels, ids)
        with self.assertRaises(ValueError):
            apply_label_consistency_gate(np.array([[1.0, np.nan]]), np.array([1]), good, [1, 2], 0.5)

    def test_budget_and_audit_are_separate_from_gate(self):
        labels = np.repeat(np.arange(5, dtype=np.int64), 100)
        rows = np.arange(1000, 1500, dtype=np.int64)
        random = apply_label_flip(labels, 0.2, 42, mode="random", allowed_class_ids=list(range(5)), original_row_indices=rows)
        targeted = apply_label_flip(labels, 0.2, 42, mode="targeted", source_class_ids=[1, 2, 3, 4], target_class_id=0, allowed_class_ids=list(range(5)), original_row_indices=rows)
        self.assertEqual((random.changed_count, random.eligible_count), (100, 500))
        self.assertEqual((targeted.changed_count, targeted.eligible_count), (80, 400))
        self.assertEqual(targeted.achieved_total_fraction, 0.16)
        np.testing.assert_array_equal(targeted.changed_original_row_indices, rows[targeted.changed_indices])
        retained = np.ones(500, dtype=bool)
        retained[targeted.changed_indices[:10]] = False
        clean_indices = np.setdiff1d(np.arange(500), targeted.changed_indices)
        retained[clean_indices[:5]] = False
        audit = replay_gate_metrics(retained, targeted.changed_indices, labels, targeted.poisoned_labels, class_ids=list(range(5)))
        self.assertEqual(audit["poisoned_candidates"], 80)
        self.assertEqual(audit["poison_rejected"], 10)
        self.assertEqual(audit["retained_poison_fraction"], 70 / 80)
        self.assertEqual(audit["clean_false_rejected"], 5)
        self.assertEqual(audit["clean_false_rejection_rate"], 5 / 420)
        self.assertEqual(audit["retained_total"], 485)
        clean = replay_gate_metrics(np.ones(500, dtype=bool), np.empty(0, dtype=np.int64), labels, labels, class_ids=list(range(5)))
        self.assertIsNone(clean["poison_rejection_rate"])
        self.assertEqual(clean["clean_false_rejection_rate"], 0.0)
        with self.assertRaises(ValueError):
            replay_gate_metrics(retained, np.array([0]), labels, targeted.poisoned_labels, class_ids=list(range(5)))
        np.testing.assert_array_equal(labels, np.repeat(np.arange(5), 100))


if __name__ == "__main__":
    unittest.main()
