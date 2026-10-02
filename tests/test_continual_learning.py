"""Synthetic, dataset-free checks for task boundaries and clean replay."""

import unittest

import numpy as np

from src.continual_learning import (
    TASK1_CLASS_IDS, TASK2_CLASS_IDS, prepare_two_task_dataset,
    select_balanced_replay,
)
from src.data import BROAD_CLASS_ORDER, SelectedRows, stratified_split_indices
from src.evaluation import forgetting_metrics


def fixture():
    labels = np.repeat(np.asarray(BROAD_CLASS_ORDER, dtype=object), 10)
    features = np.arange(80, dtype=np.float64).reshape(-1, 1) + 1
    features[50:] += 10000  # future classes must not affect fitted statistics
    rows = np.arange(1000, 1080, dtype=np.int64)
    selected = SelectedRows(features, labels, rows, ("flow_feature",))
    split = stratified_split_indices(labels, seed=42)
    return selected, split


class ContinualTaskTests(unittest.TestCase):
    def test_membership_global_ids_disjoint_rows_and_frozen_fit(self):
        selected, split = fixture()
        selected.features[split.validation] += 1_000_000
        selected.features[split.test] += 1_000_000
        original_features = selected.features.copy()
        tasks = prepare_two_task_dataset(selected, split)
        self.assertEqual(tasks.task1.class_ids, TASK1_CLASS_IDS)
        self.assertEqual(tasks.task2.class_ids, TASK2_CLASS_IDS)
        self.assertEqual(tasks.class_names, dict(enumerate(BROAD_CLASS_ORDER)))
        for name in ("train", "validation", "test"):
            old = getattr(tasks.task1, name)
            new = getattr(tasks.task2, name)
            self.assertEqual(set(np.unique(old.labels)), set(TASK1_CLASS_IDS))
            self.assertEqual(set(np.unique(new.labels)), set(TASK2_CLASS_IDS))
            self.assertFalse(np.intersect1d(old.row_indices, new.row_indices).size)
        train_rows = np.concatenate((tasks.task1.train.row_indices, tasks.task2.train.row_indices))
        validation_rows = tasks.seen_validation.row_indices
        test_rows = tasks.combined_test.row_indices
        self.assertFalse(np.intersect1d(train_rows, validation_rows).size)
        self.assertFalse(np.intersect1d(train_rows, test_rows).size)
        self.assertFalse(np.intersect1d(validation_rows, test_rows).size)
        np.testing.assert_array_equal(np.sort(np.concatenate((train_rows, validation_rows, test_rows))), selected.row_indices)
        task1_train_raw = selected.features[split.train[np.isin(selected.labels[split.train], BROAD_CLASS_ORDER[:5])]]
        self.assertAlmostEqual(tasks.preprocessor.means[0], task1_train_raw.mean())
        self.assertLess(tasks.preprocessor.means[0], 100)
        np.testing.assert_allclose(tasks.task2.train.features,
                                   tasks.preprocessor.transform(selected.features[split.train[np.isin(selected.labels[split.train], BROAD_CLASS_ORDER[5:])]]))
        np.testing.assert_array_equal(selected.features, original_features)

    def test_replay_balance_determinism_copies_and_exclusions(self):
        selected, split = fixture()
        tasks = prepare_two_task_dataset(selected, split)
        forbidden = np.concatenate((tasks.seen_validation.row_indices, tasks.combined_test.row_indices))
        first = select_balanced_replay(tasks.task1.train, class_ids=TASK1_CLASS_IDS,
                                       per_class=2, seed=42, forbidden_row_indices=forbidden)
        second = select_balanced_replay(tasks.task1.train, class_ids=TASK1_CLASS_IDS,
                                        per_class=2, seed=42, forbidden_row_indices=forbidden)
        self.assertEqual(first.class_counts, {class_id: 2 for class_id in TASK1_CLASS_IDS})
        self.assertEqual(len(first.row_indices), 10)
        self.assertEqual(len(np.unique(first.row_indices)), 10)
        self.assertFalse(np.intersect1d(first.row_indices, forbidden).size)
        np.testing.assert_array_equal(first.row_indices, second.row_indices)
        X_replay, y_replay = first.as_fit_replay()
        self.assertEqual((X_replay.shape, y_replay.shape), ((10, 1), (10,)))
        X_replay[0, 0] = -999
        self.assertNotEqual(first.features[0, 0], -999)
        with self.assertRaises(ValueError):
            select_balanced_replay(tasks.task1.train, class_ids=TASK1_CLASS_IDS,
                                   per_class=8, seed=42)
        with self.assertRaises(ValueError):
            select_balanced_replay(tasks.task1.train, class_ids=TASK1_CLASS_IDS,
                                   per_class=2, seed=42,
                                   forbidden_row_indices=[int(tasks.task1.train.row_indices[0])])

    def test_rejects_overlapping_split(self):
        selected, split = fixture()
        split.validation[0] = split.train[0]
        with self.assertRaises(ValueError):
            prepare_two_task_dataset(selected, split)


class ForgettingTests(unittest.TestCase):
    def test_fixed_old_macro_and_new_prediction_errors(self):
        truth = np.array([0, 0, 1, 1])
        before = np.array([0, 0, 1, 1])
        after = np.array([0, 5, 1, 5])
        result = forgetting_metrics(truth, before, after,
                                    old_class_ids=[0, 1], all_class_ids=[0, 1, 5])
        self.assertEqual(result["accuracy_forgetting"], 0.5)
        self.assertEqual(result["accuracy_change"], -0.5)
        self.assertAlmostEqual(result["macro_f1_before"], 1.0)
        self.assertAlmostEqual(result["macro_f1_after"], 2 / 3)
        self.assertAlmostEqual(result["macro_f1_forgetting"], 1 / 3)
        improved = forgetting_metrics(truth, after, before,
                                      old_class_ids=[0, 1], all_class_ids=[0, 1, 5])
        self.assertEqual(improved["accuracy_forgetting"], -0.5)
        self.assertEqual(improved["accuracy_change"], 0.5)


if __name__ == "__main__":
    unittest.main()
