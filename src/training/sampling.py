"""Deterministic class-weight plan without importing PyTorch or data files."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SamplingWeights:
    row_weights: np.ndarray
    class_counts: dict[int, int]
    expected_class_probabilities: dict[int, float]


def inverse_frequency_weights(supplied_labels: np.ndarray) -> SamplingWeights:
    """Give each row weight ``1 / count(its supplied class)``.

    The plan does not draw or copy feature rows. With replacement, these
    weights imply equal *expected* probability for every class present, not
    exact per-epoch balance or equal unique-example exposure.
    """

    labels = np.asarray(supplied_labels)
    if labels.ndim != 1 or labels.size == 0 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("supplied_labels must be a nonempty one-dimensional integer array")
    ids, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float64)
    total_weight = float(weights.sum())
    return SamplingWeights(
        row_weights=weights.copy(),
        class_counts={int(class_id): int(count) for class_id, count in zip(ids, counts)},
        expected_class_probabilities={
            int(class_id): float(weights[labels == class_id].sum() / total_weight)
            for class_id in ids
        },
    )
