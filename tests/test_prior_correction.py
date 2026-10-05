"""Synthetic, prediction-only checks; no research training or test tuning."""

import numpy as np
import pytest

from src.evaluation.prior_correction import (
    STRENGTHS, corrected_predictions, prior_log_shift, select_candidate,
    summarize_metrics,
)
from src.evaluation.streaming import ConfusionAccumulator


def test_zero_control_and_direction_of_training_prior_shift():
    original = np.array([80, 5, 4, 3, 2, 3, 2, 1])
    sampled = np.array([0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    shift = prior_log_shift(original, sampled)
    assert shift[0] > shift[5]  # Raise Benign relative to oversampled Infiltration.
    logits = np.array([[0, 0, 0, 0, 0, 0.2, 0, 0]], dtype=np.float32)
    assert corrected_predictions(logits, shift, 0.0).tolist() == [5]
    assert corrected_predictions(logits, shift, 1.0).tolist() == [0]
    with pytest.raises(ValueError):
        corrected_predictions(logits, shift, 1.5)
    with pytest.raises(ValueError):
        prior_log_shift(original, np.ones(8) / 9)
    assert STRENGTHS == (0.0, 0.25, 0.5, 0.75, 1.0)


def test_locked_acceptance_and_selection_no_eligible():
    accumulator = ConfusionAccumulator(tuple(range(8)))
    truth = np.array([0] * 20 + list(range(1, 8)))
    predictions = truth.copy()
    predictions[0] = 5  # Exactly 5% Benign FPR is eligible.
    accumulator.update(truth, predictions)
    metrics = accumulator.metrics(class_names={i: str(i) for i in range(8)}, benign_class_id=0)
    summary = summarize_metrics(metrics)
    assert summary["benign_fpr"] == 0.05
    assert summary["old_accuracy"] >= 0.90
    assert summary["new_mean_recall"] == 1.0
    assert summary["eligible"]
    winner = {"arm": "A", "strength": 0.25, "summary": summary}
    assert select_candidate([winner]) == winner
    failure = {**winner, "summary": {**summary, "eligible": False}}
    assert select_candidate([failure]) is None
    with pytest.raises(ValueError):
        summarize_metrics({"confusion_matrix": [[1]], "per_class": {}})


def test_selection_uses_eligible_validation_macro_f1_only():
    def row(arm, strength, f1, eligible):
        return {"arm": arm, "strength": strength,
                "summary": {"macro_f1": f1, "benign_fpr": 0.03,
                            "new_mean_recall": 0.7, "eligible": eligible}}
    chosen = select_candidate([
        row("A", 0.0, 0.99, False),
        row("B", 0.5, 0.81, True),
        row("A", 0.75, 0.82, True),
    ])
    assert (chosen["arm"], chosen["strength"]) == ("A", 0.75)
