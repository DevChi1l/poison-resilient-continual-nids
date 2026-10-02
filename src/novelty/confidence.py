"""NumPy-only confidence baseline for marking predictions as unknown.

This module treats the largest probability in each model-output row as the
classifier's confidence. A row is unknown only when that confidence is
strictly below the configured threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class NoveltyResult:
    """Prediction details aligned row-for-row with a probability matrix."""

    predicted_class_ids: np.ndarray
    maximum_probabilities: np.ndarray
    is_unknown: np.ndarray


def _validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    array = np.asarray(probabilities)
    if array.ndim != 2:
        raise ValueError("probabilities must have shape (n_samples, n_classes)")
    if array.shape[1] == 0:
        raise ValueError("probabilities must contain at least one class column")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError("probabilities must contain real numeric values")
    if not np.isfinite(array).all():
        raise ValueError("probabilities must contain only finite values")
    if ((array < 0) | (array > 1)).any():
        raise ValueError("probabilities must be between 0 and 1")
    if not np.allclose(array.sum(axis=1), 1.0, rtol=1e-5, atol=1e-8):
        raise ValueError("each probability row must sum approximately to 1")
    return array


def _validate_threshold(threshold: float) -> float:
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real):
        raise ValueError("threshold must be a finite real scalar")
    normalized = float(threshold)
    if not np.isfinite(normalized):
        raise ValueError("threshold must be finite")
    if not 0.0 <= normalized <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    return normalized


def detect_unknown(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Return a boolean mask for rows whose maximum probability is too low.

    Confidence equal to ``threshold`` is known because the comparison is
    strictly less-than. The input probability matrix is never modified.
    """

    array = _validate_probabilities(probabilities)
    normalized_threshold = _validate_threshold(threshold)
    return np.max(array, axis=1) < normalized_threshold


def summarize_predictions(
    probabilities: np.ndarray,
    class_ids: Sequence[int],
    threshold: float,
) -> NoveltyResult:
    """Map output columns to external class IDs and attach novelty decisions."""

    array = _validate_probabilities(probabilities)
    normalized_threshold = _validate_threshold(threshold)
    external_ids = np.asarray(class_ids)
    if external_ids.ndim != 1:
        raise ValueError("class_ids must be one-dimensional")
    if external_ids.shape[0] != array.shape[1]:
        raise ValueError("class_ids length must match probability columns")
    if not np.issubdtype(external_ids.dtype, np.integer):
        raise ValueError("class_ids must contain integer external class IDs")
    if np.unique(external_ids).size != external_ids.size:
        raise ValueError("class_ids must be unique")

    maximum_probabilities = np.max(array, axis=1)
    predicted_class_ids = external_ids[np.argmax(array, axis=1)]
    return NoveltyResult(
        predicted_class_ids=predicted_class_ids.copy(),
        maximum_probabilities=maximum_probabilities,
        is_unknown=maximum_probabilities < normalized_threshold,
    )
