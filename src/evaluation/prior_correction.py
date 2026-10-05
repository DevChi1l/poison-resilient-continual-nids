"""Validation-only log-prior correction for fixed eight-class checkpoints.

This is a prediction adjustment, not model fitting or probability calibration.
The sampling-prior assumption is exploratory for a continually trained model.
"""

from __future__ import annotations

import numpy as np


STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
TARGETS = {"benign_fpr_max": 0.05, "old_accuracy_min": 0.90,
           "new_mean_recall_min": 0.50}


def prior_log_shift(original_train_counts: np.ndarray,
                    sampled_probabilities: np.ndarray) -> np.ndarray:
    """Compute log(P_original_train / P_training_draw), in class-ID order."""
    counts = np.asarray(original_train_counts, dtype=np.float64)
    sampled = np.asarray(sampled_probabilities, dtype=np.float64)
    if (counts.shape != (8,) or sampled.shape != (8,) or
            not np.isfinite(counts).all() or not np.isfinite(sampled).all() or
            np.any(counts <= 0) or np.any(sampled <= 0) or
            not np.isclose(sampled.sum(), 1.0, rtol=0, atol=1e-12)):
        raise ValueError("Require eight positive original counts and eight draw probabilities")
    return np.log(counts / counts.sum()) - np.log(sampled)


def corrected_predictions(logits: np.ndarray, shift: np.ndarray,
                          strength: float) -> np.ndarray:
    """Return external IDs 0..7; zero strength exactly matches raw argmax."""
    scores = np.asarray(logits)
    offsets = np.asarray(shift)
    if (scores.ndim != 2 or scores.shape[1] != 8 or offsets.shape != (8,) or
            not np.isfinite(scores).all() or not np.isfinite(offsets).all() or
            strength not in STRENGTHS):
        raise ValueError("Require finite eight-class logits/shift and locked strength")
    return np.argmax(scores + strength * offsets, axis=1).astype(np.int64)


def summarize_metrics(metrics: dict) -> dict:
    """Expose acceptance quantities without hiding minority-class metrics."""
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    if matrix.shape != (8, 8) or np.any(matrix < 0):
        raise ValueError("Expected eight-class nonnegative confusion counts")
    old_rows, new_rows = int(matrix[:5].sum()), int(matrix[5:].sum())
    if old_rows == 0 or new_rows == 0 or int(matrix[0].sum()) == 0:
        raise ValueError("Old, new, and Benign validation support must be nonzero")
    old_accuracy = float(np.trace(matrix[:5, :5]) / old_rows)
    new_accuracy = float(np.trace(matrix[5:, 5:]) / new_rows)
    new_mean_recall = float(np.mean([
        metrics["per_class"][str(class_id)]["recall"] for class_id in (5, 6, 7)
    ]))
    benign_fpr = float(metrics["benign_false_positive"]["rate"])
    return {"accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "benign_fpr": benign_fpr, "old_accuracy": old_accuracy,
            "new_accuracy": new_accuracy, "new_mean_recall": new_mean_recall,
            "eligible": (benign_fpr <= TARGETS["benign_fpr_max"] and
                         old_accuracy >= TARGETS["old_accuracy_min"] and
                         new_mean_recall >= TARGETS["new_mean_recall_min"])}


def select_candidate(rows: list[dict]) -> dict | None:
    """Highest eligible validation macro-F1; stable tie-breaks are predeclared."""
    eligible = [row for row in rows if row["summary"]["eligible"]]
    if not eligible:
        return None
    return min(eligible, key=lambda row: (
        -row["summary"]["macro_f1"], row["summary"]["benign_fpr"],
        -row["summary"]["new_mean_recall"], row["arm"], row["strength"],
    ))
