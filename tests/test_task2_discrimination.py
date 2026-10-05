"""Synthetic protocol tests; no research-data training."""

import numpy as np

from scripts import run_task2_discrimination as study
from src.evaluation.streaming import ConfusionAccumulator


class TinyOldTrain:
    partition = "train"
    class_ids = tuple(range(5))

    def __init__(self):
        self.labels = np.repeat(np.arange(5), 5)
        self.splits = np.tile(np.array([0, 0, 0, 0, 2], dtype=np.uint8), 5)
        self.indices = np.flatnonzero(self.splits == 0)
        self.calls = []

    def batch(self, positions):
        self.calls.append(len(positions))
        values = np.asarray(positions, dtype=np.float32)
        return np.column_stack((values, values + 1)), self.labels[positions].copy(), positions.copy()


def test_replay_is_distinct_training_only_and_loaded_in_batches(monkeypatch):
    monkeypatch.setattr(study, "REPLAY_COUNTS", {i: 2 for i in range(5)})
    view = TinyOldTrain()
    positions = study.select_replay_positions(view)
    assert len(positions) == len(np.unique(positions)) == 10
    assert np.all(view.splits[positions] == 0)
    X, labels, ids = study.replay_batches(view, positions, width=2)
    np.testing.assert_array_equal(ids, positions)
    assert X.shape == (10, 2) and np.isfinite(X).all()
    assert view.calls == [10]
    assert np.bincount(labels, minlength=5).tolist() == [2] * 5
    view.partition = "validation"
    try:
        study.select_replay_positions(view)
    except ValueError:
        pass
    else:
        raise AssertionError("Non-training replay source was accepted")


def test_schedules_are_locked_and_only_sampling_changes():
    names = tuple(str(i) for i in range(8))
    split = {"counts": {str(i): {"train": 100 * (i + 1)} for i in range(5)}}
    a = study.schedule(split, names, 0.70)
    b = study.schedule(split, names, 0.75)
    assert np.isclose(sum(a.values()), 1) and np.isclose(sum(b.values()), 1)
    assert np.isclose(sum(a[i] for i in range(5)), 0.70)
    assert np.isclose(sum(b[i] for i in range(5)), 0.75)
    assert np.allclose((a[5], a[6], a[7]), (0.20, 0.05, 0.05))
    assert np.allclose((b[5], b[6], b[7]), (0.15, 0.05, 0.05))
    assert study.DRAWS == 211_728 and study.MAX_EPOCHS == 12
    try:
        study.schedule(split, names, 0.80)
    except ValueError:
        pass
    else:
        raise AssertionError("Unplanned schedule was accepted")


def test_each_new_class_recall_is_required_individually():
    accumulator = ConfusionAccumulator(tuple(range(8)))
    truth = np.array([0] * 20 + list(range(1, 8)))
    predicted = truth.copy()
    predicted[-3] = 0  # Infiltration missed; other new recalls are perfect.
    accumulator.update(truth, predicted)
    metrics = accumulator.metrics(class_names={i: str(i) for i in range(8)},
                                  benign_class_id=0)
    summary = study.summarize(metrics)
    assert summary["benign_fpr"] == 0
    assert summary["old_accuracy"] == 1
    assert summary["new_recall"] == {"5": 0, "6": 1, "7": 1}
    assert not summary["eligible"]
    records = {name: {"summary": summary} for name in study.ARMS}
    assert study.select(records) is None
    predicted[-3] = 5
    accumulator = ConfusionAccumulator(tuple(range(8)))
    accumulator.update(truth, predicted)
    eligible = study.summarize(accumulator.metrics(
        class_names={i: str(i) for i in range(8)}, benign_class_id=0))
    assert eligible["eligible"]
    records[study.ARMS[0]] = {"summary": eligible}
    assert study.select(records) == study.ARMS[0]
