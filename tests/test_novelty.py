"""Synthetic tests for the NumPy-only confidence novelty baseline."""

from __future__ import annotations

import unittest

import numpy as np

from src.novelty import detect_unknown, summarize_predictions


class ConfidenceNoveltyTests(unittest.TestCase):
    def test_marks_known_and_unknown_predictions(self) -> None:
        probabilities = np.asarray(
            [
                [0.90, 0.10],
                [0.55, 0.45],
            ],
            dtype=np.float32,
        )

        result = detect_unknown(probabilities, threshold=0.60)

        np.testing.assert_array_equal(result, np.asarray([False, True]))
        self.assertEqual(result.dtype, np.bool_)

    def test_confidence_equal_to_threshold_is_known(self) -> None:
        probabilities = np.asarray([[0.60, 0.40]], dtype=np.float64)

        np.testing.assert_array_equal(
            detect_unknown(probabilities, threshold=0.60),
            np.asarray([False]),
        )

    def test_threshold_endpoints(self) -> None:
        probabilities = np.asarray([[1.0, 0.0], [0.5, 0.5]])

        np.testing.assert_array_equal(
            detect_unknown(probabilities, threshold=0.0),
            np.asarray([False, False]),
        )
        np.testing.assert_array_equal(
            detect_unknown(probabilities, threshold=1.0),
            np.asarray([False, True]),
        )

    def test_empty_batch_returns_shaped_empty_outputs(self) -> None:
        probabilities = np.empty((0, 3), dtype=np.float32)

        unknown = detect_unknown(probabilities, threshold=0.5)
        summary = summarize_predictions(probabilities, [10, 30, 90], threshold=0.5)

        self.assertEqual(unknown.shape, (0,))
        self.assertEqual(unknown.dtype, np.bool_)
        self.assertEqual(summary.predicted_class_ids.shape, (0,))
        self.assertEqual(summary.maximum_probabilities.shape, (0,))
        self.assertEqual(summary.is_unknown.shape, (0,))

    def test_helper_uses_non_contiguous_external_class_ids(self) -> None:
        probabilities = np.asarray([[0.10, 0.70, 0.20], [0.45, 0.25, 0.30]])

        result = summarize_predictions(
            probabilities,
            class_ids=[10, 30, 90],
            threshold=0.50,
        )

        np.testing.assert_array_equal(result.predicted_class_ids, [30, 10])
        np.testing.assert_allclose(result.maximum_probabilities, [0.70, 0.45])
        np.testing.assert_array_equal(result.is_unknown, [False, True])

    def test_does_not_modify_probability_input(self) -> None:
        probabilities = np.asarray([[0.75, 0.25], [0.40, 0.60]])
        original = probabilities.copy()

        detect_unknown(probabilities, threshold=0.70)
        summarize_predictions(probabilities, [4, 8], threshold=0.70)

        np.testing.assert_array_equal(probabilities, original)

    def test_rejects_invalid_probability_inputs(self) -> None:
        invalid_inputs = (
            np.asarray([0.5, 0.5]),
            np.empty((2, 0)),
            np.asarray([[np.nan, np.nan]]),
            np.asarray([[np.inf, 0.0]]),
            np.asarray([[-0.1, 1.1]]),
            np.asarray([[0.4, 0.4]]),
            np.asarray([["0.5", "0.5"]]),
            np.asarray([[0.5 + 0j, 0.5 + 0j]]),
        )

        for probabilities in invalid_inputs:
            with self.subTest(probabilities=probabilities):
                with self.assertRaises(ValueError):
                    detect_unknown(probabilities, threshold=0.5)

    def test_rejects_invalid_thresholds(self) -> None:
        probabilities = np.asarray([[0.5, 0.5]])

        for threshold in (-0.01, 1.01, np.nan, np.inf, True, "0.5", [0.5]):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    detect_unknown(probabilities, threshold=threshold)  # type: ignore[arg-type]

    def test_helper_rejects_invalid_class_ids(self) -> None:
        probabilities = np.asarray([[0.2, 0.8]])

        for class_ids in ([10], [10, 10], [10.0, 20.0], [[10, 20]]):
            with self.subTest(class_ids=class_ids):
                with self.assertRaises(ValueError):
                    summarize_predictions(probabilities, class_ids, threshold=0.5)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
