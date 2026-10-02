"""JSON-serializable known-class classification and poisoning diagnostics."""

from __future__ import annotations

from numbers import Integral
from typing import Mapping, Sequence

import numpy as np


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_ids: Sequence[int] | None = None,
    class_names: Mapping[int, str] | None = None,
    benign_class_id: int | None = None,
    source_class_ids: Sequence[int] | None = None,
    target_class_id: int | None = None,
) -> dict:
    """Return known-class metrics and optional benign/source-target error rates.

    The source-target rate measures predictions of the target among true source
    rows. It is an observed error rate, not proof that poisoning caused errors.
    """

    truth = np.asarray(y_true)
    predictions = np.asarray(y_pred)
    if (
        truth.ndim != 1
        or predictions.ndim != 1
        or truth.size == 0
        or truth.shape != predictions.shape
        or not np.issubdtype(truth.dtype, np.integer)
        or not np.issubdtype(predictions.dtype, np.integer)
    ):
        raise ValueError("y_true and y_pred must be nonempty, aligned integer arrays")
    ids = np.unique(np.concatenate((truth, predictions))) if class_ids is None else np.asarray(class_ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or ids.size == 0:
        raise ValueError("class_ids must be a nonempty one-dimensional integer array")
    if np.unique(ids).size != ids.size or not np.isin(truth, ids).all() or not np.isin(predictions, ids).all():
        raise ValueError("class_ids must be unique and cover all labels and predictions")

    confusion = np.asarray(
        [[np.count_nonzero((truth == actual) & (predictions == predicted)) for predicted in ids]
         for actual in ids],
        dtype=np.int64,
    )
    per_class = {}
    for index, class_id in enumerate(ids):
        tp = int(confusion[index, index])
        predicted_count = int(confusion[:, index].sum())
        support = int(confusion[index, :].sum())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        normalized_id = int(class_id)
        per_class[str(normalized_id)] = {
            "name": class_names[normalized_id] if class_names is not None else str(normalized_id),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    result = {
        "accuracy": float(np.trace(confusion) / truth.size),
        "macro_f1": float(np.mean([item["f1"] for item in per_class.values()])),
        "class_ids": [int(value) for value in ids],
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }
    if benign_class_id is not None:
        if isinstance(benign_class_id, (bool, np.bool_)) or not isinstance(benign_class_id, Integral) or benign_class_id not in ids:
            raise ValueError("benign_class_id must be a class ID")
        benign_rows = truth == benign_class_id
        support = int(benign_rows.sum())
        errors = int(np.count_nonzero(benign_rows & (predictions != benign_class_id)))
        result["benign_false_positive"] = {
            "class_id": int(benign_class_id),
            "count": errors,
            "support": support,
            "rate": errors / support if support else 0.0,
        }
    if (source_class_ids is None) != (target_class_id is None):
        raise ValueError("source_class_ids and target_class_id must be supplied together")
    if source_class_ids is not None:
        sources = np.asarray(source_class_ids)
        if (
            sources.ndim != 1
            or not np.issubdtype(sources.dtype, np.integer)
            or sources.size == 0
            or np.unique(sources).size != sources.size
            or not np.isin(sources, ids).all()
        ):
            raise ValueError("source_class_ids must be unique class IDs")
        if isinstance(target_class_id, (bool, np.bool_)) or not isinstance(target_class_id, Integral) or target_class_id not in ids or target_class_id in sources:
            raise ValueError("target_class_id must be a different class ID")
        source_rows = np.isin(truth, sources)
        support = int(source_rows.sum())
        errors = int(np.count_nonzero(source_rows & (predictions == target_class_id)))
        result["source_to_target_misclassification"] = {
            "source_class_ids": [int(value) for value in sources],
            "target_class_id": int(target_class_id),
            "count": errors,
            "support": support,
            "rate": errors / support if support else 0.0,
        }
    return result
