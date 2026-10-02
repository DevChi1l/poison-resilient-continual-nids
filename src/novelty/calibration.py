"""Known-validation-only calibration for the maximum-confidence baseline."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np

from .confidence import _validate_probabilities


@dataclass(frozen=True)
class ConfidenceCalibration:
    threshold: float
    quantile: float
    validation_count: int
    method: str = "linear"


def calibrate_confidence_threshold(
    known_validation_probabilities: np.ndarray,
    *,
    quantile: float = 0.05,
) -> ConfidenceCalibration:
    """Use the fifth percentile of known maximum probabilities by default.

    Only known-class validation rows may be passed. The caller must freeze
    this threshold before evaluating genuinely unseen classes. A zero-row
    validation matrix is not calibratable.
    """

    if isinstance(quantile, (bool, np.bool_)) or not isinstance(quantile, Real):
        raise ValueError("quantile must be a finite scalar in [0, 1]")
    normalized = float(quantile)
    if not np.isfinite(normalized) or not 0 <= normalized <= 1:
        raise ValueError("quantile must be a finite scalar in [0, 1]")
    probabilities = _validate_probabilities(known_validation_probabilities)
    if probabilities.shape[0] == 0:
        raise ValueError("known validation must contain at least one row")
    confidences = np.max(probabilities, axis=1)
    return ConfidenceCalibration(
        threshold=float(np.quantile(confidences, normalized, method="linear")),
        quantile=normalized,
        validation_count=int(confidences.size),
    )
