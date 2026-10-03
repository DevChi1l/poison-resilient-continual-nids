"""NumPy-only replay-label consistency scoring against a frozen old teacher."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class LabelConsistencyCalibration:
    threshold: float
    quantile: float
    validation_count: int
    method: str = "linear"


@dataclass(frozen=True)
class GateResult:
    retained_features: np.ndarray
    retained_labels: np.ndarray
    retained_mask: np.ndarray
    scores: np.ndarray
    threshold: float

    def as_fit_replay(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return copied model replay arrays, or ``None`` when none remain."""

        if self.retained_labels.size == 0:
            return None
        return self.retained_features.copy(), self.retained_labels.copy()


def _validate_probability_mapping(
    probabilities: np.ndarray,
    supplied_labels: np.ndarray,
    class_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = np.asarray(probabilities)
    labels = np.asarray(supplied_labels)
    ids = np.asarray(class_ids)
    if (probs.ndim != 2 or probs.shape[1] == 0 or
            not np.issubdtype(probs.dtype, np.number) or np.iscomplexobj(probs) or
            not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)) or
            not np.allclose(probs.sum(axis=1), 1.0, rtol=1e-5, atol=1e-8)):
        raise ValueError("teacher probabilities must be finite, normalized rows")
    if (ids.ndim != 1 or ids.size != probs.shape[1] or
            not np.issubdtype(ids.dtype, np.integer) or np.unique(ids).size != ids.size):
        raise ValueError("class_ids must uniquely map teacher probability columns")
    if (labels.ndim != 1 or labels.size != probs.shape[0] or
            not np.issubdtype(labels.dtype, np.integer) or not np.isin(labels, ids).all()):
        raise ValueError("supplied labels must align with rows and teacher class_ids")
    return probs, labels, ids


def _unit_scalar(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite scalar in [0, 1]")
    number = float(value)
    if not np.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{name} must be a finite scalar in [0, 1]")
    return number


def label_inconsistency_scores(
    teacher_probabilities: np.ndarray,
    supplied_labels: np.ndarray,
    class_ids: Sequence[int],
) -> np.ndarray:
    """Return ``1 - p_teacher(supplied_label)`` using explicit external IDs."""

    probs, labels, ids = _validate_probability_mapping(
        teacher_probabilities, supplied_labels, class_ids
    )
    column_by_id = {int(class_id): column for column, class_id in enumerate(ids)}
    columns = np.fromiter((column_by_id[int(label)] for label in labels),
                          dtype=np.int64, count=len(labels))
    return (1.0 - probs[np.arange(len(labels)), columns]).copy()


def calibrate_label_consistency(
    clean_validation_probabilities: np.ndarray,
    clean_validation_labels: np.ndarray,
    class_ids: Sequence[int],
    *,
    quantile: float = 0.95,
) -> LabelConsistencyCalibration:
    """Freeze a linear-interpolation quantile from clean old-class validation."""

    scores = label_inconsistency_scores(
        clean_validation_probabilities, clean_validation_labels, class_ids
    )
    return calibrate_label_consistency_scores(scores, quantile=quantile)


def calibrate_label_consistency_scores(
    clean_validation_scores: np.ndarray, *, quantile: float = 0.95,
) -> LabelConsistencyCalibration:
    """Calibrate from bounded-batch scores retained for clean old validation.

    This accepts only scalar inconsistency scores, not future-class examples
    or simulator poison masks. The caller must establish the fit scope.
    """

    normalized_quantile = _unit_scalar(quantile, "quantile")
    scores = np.asarray(clean_validation_scores)
    if (scores.ndim != 1 or not np.issubdtype(scores.dtype, np.number) or
            np.iscomplexobj(scores) or not np.isfinite(scores).all() or
            np.any((scores < 0) | (scores > 1))):
        raise ValueError("clean validation scores must be finite values in [0, 1]")
    if scores.size == 0:
        raise ValueError("clean Task 1 validation must contain at least one row")
    return LabelConsistencyCalibration(
        threshold=float(np.quantile(scores, normalized_quantile, method="linear")),
        quantile=normalized_quantile,
        validation_count=int(scores.size),
    )


def apply_label_consistency_gate(
    features: np.ndarray,
    supplied_labels: np.ndarray,
    teacher_probabilities: np.ndarray,
    class_ids: Sequence[int],
    threshold: float,
) -> GateResult:
    """Quarantine scores above threshold without seeing true labels or attack masks.

    This function has no parameter for original labels or changed indices.
    It is only appropriate for old-class replay scored by the frozen clean
    Task 1 teacher; callers must never use it on new Task 2 classes.
    """

    normalized_threshold = _unit_scalar(threshold, "threshold")
    X = np.asarray(features)
    if X.ndim != 2 or not np.issubdtype(X.dtype, np.number) or np.iscomplexobj(X) or not np.isfinite(X).all():
        raise ValueError("features must be a finite two-dimensional numeric matrix")
    scores = label_inconsistency_scores(teacher_probabilities, supplied_labels, class_ids)
    if len(X) != len(scores):
        raise ValueError("features and supplied labels must have the same row count")
    retained = scores <= normalized_threshold
    return GateResult(
        retained_features=X[retained].copy(),
        retained_labels=np.asarray(supplied_labels)[retained].copy(),
        retained_mask=retained.copy(),
        scores=scores,
        threshold=normalized_threshold,
    )
