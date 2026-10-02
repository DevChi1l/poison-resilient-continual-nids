"""Binary held-out unknown detection metrics with explicit edge cases."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def novelty_metrics(
    is_unknown_true: np.ndarray,
    is_unknown_pred: np.ndarray,
    *,
    true_class_ids: np.ndarray | None = None,
    unknown_class_ids: Sequence[int] | None = None,
) -> dict:
    """Summarize unknown flags, optionally by external held-out class ID.

    Zero predicted unknowns yield precision 0. If no actual unknowns exist,
    recall and F1 are undefined (``None``); likewise known false rejection
    is undefined when no known rows exist. No threshold selection occurs here.
    """

    truth = np.asarray(is_unknown_true)
    predicted = np.asarray(is_unknown_pred)
    if (truth.ndim != 1 or predicted.shape != truth.shape or
            truth.dtype != np.bool_ or predicted.dtype != np.bool_):
        raise ValueError("unknown truth and prediction must be aligned boolean arrays")
    true_ids = None
    unknown_ids = None
    if (true_class_ids is None) != (unknown_class_ids is None):
        raise ValueError("true_class_ids and unknown_class_ids must be supplied together")
    if true_class_ids is not None:
        true_ids = np.asarray(true_class_ids)
        unknown_ids = np.asarray(unknown_class_ids)
        if (true_ids.shape != truth.shape or not np.issubdtype(true_ids.dtype, np.integer) or
                unknown_ids.ndim != 1 or unknown_ids.size == 0 or
                not np.issubdtype(unknown_ids.dtype, np.integer) or
                np.unique(unknown_ids).size != unknown_ids.size or
                not np.array_equal(np.isin(true_ids, unknown_ids), truth)):
            raise ValueError("external unknown class IDs must exactly match the truth flags")
    unknown_count = int(truth.sum())
    known_count = int((~truth).sum())
    tp = int(np.count_nonzero(truth & predicted))
    fp = int(np.count_nonzero(~truth & predicted))
    fn = unknown_count - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / unknown_count if unknown_count else None
    f1 = (2 * precision * recall / (precision + recall)
          if recall is not None and precision + recall else
          (0.0 if recall is not None else None))
    per_class = {}
    if true_ids is not None:
        for class_id in unknown_ids:
            class_rows = true_ids == class_id
            support = int(class_rows.sum())
            flagged = int(np.count_nonzero(class_rows & predicted))
            per_class[str(int(class_id))] = {
                "support": support,
                "flagged_unknown": flagged,
                "recall": flagged / support if support else None,
            }
    return {
        "unknown_true": unknown_count,
        "known_true": known_count,
        "true_unknown_flagged": tp,
        "known_false_rejected": fp,
        "unknown_missed": fn,
        "unknown_precision": precision,
        "unknown_recall": recall,
        "unknown_f1": f1,
        "known_false_rejection_rate": fp / known_count if known_count else None,
        "per_unknown_class": per_class,
    }
