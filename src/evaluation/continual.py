"""Old-class forgetting with a fixed denominator and class set."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .classification import classification_metrics


def forgetting_metrics(
    old_truth: np.ndarray,
    before_predictions: np.ndarray,
    after_predictions: np.ndarray,
    *,
    old_class_ids: Sequence[int],
    all_class_ids: Sequence[int],
) -> dict:
    """Old-class before-minus-after scores; negative values mean improvement.

    Both predictions use the *same old-class test rows*. Macro-F1 always
    averages the same old classes. A prediction into a newly introduced class
    remains a false negative for its true old class, not a dropped row.
    """

    truth = np.asarray(old_truth)
    old_ids = tuple(int(value) for value in old_class_ids)
    all_ids = tuple(int(value) for value in all_class_ids)
    if (not old_ids or len(set(old_ids)) != len(old_ids) or
            len(set(all_ids)) != len(all_ids) or not set(old_ids).issubset(all_ids) or
            truth.ndim != 1 or not np.isin(truth, old_ids).all()):
        raise ValueError("old_class_ids must be unique and cover old test labels")
    before = classification_metrics(truth, before_predictions, class_ids=all_ids)
    after = classification_metrics(truth, after_predictions, class_ids=all_ids)

    def old_macro(result: dict) -> float:
        return float(np.mean([result["per_class"][str(class_id)]["f1"] for class_id in old_ids]))

    before_f1, after_f1 = old_macro(before), old_macro(after)
    return {
        "old_class_ids": list(old_ids),
        "old_test_rows": int(truth.size),
        "accuracy_before": before["accuracy"],
        "accuracy_after": after["accuracy"],
        "accuracy_change": after["accuracy"] - before["accuracy"],
        "accuracy_forgetting": before["accuracy"] - after["accuracy"],
        "macro_f1_before": before_f1,
        "macro_f1_after": after_f1,
        "macro_f1_change": after_f1 - before_f1,
        "macro_f1_forgetting": before_f1 - after_f1,
    }
