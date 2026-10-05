"""No XGBoost install or research training required for protocol smoke tests."""

from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pytest

from scripts.run_task2_xgboost_baseline import (
    EVIDENCE_SHA, class_weights, read_evidence, summarize,
)


EVIDENCE = Path(__file__).resolve().parents[1] / "data/train2_infiltration_disc/results.zip"


def test_train_only_weight_policy_is_moderate_and_mean_one():
    counts = np.asarray([100000, 10000, 10000, 10000, 10000, 60000, 6000, 4076])
    weights = class_weights(counts)
    assert np.isclose(np.dot(weights, counts / counts.sum()), 1.0)
    assert np.all(weights > 0)
    assert max(weights) / min(weights) <= 5.0
    with pytest.raises(ValueError, match="All eight"):
        class_weights(np.asarray([1, 1, 1, 1, 1, 1, 1, 0]))


@pytest.mark.skipif(not EVIDENCE.is_file(), reason="private original result ZIP unavailable")
def test_verified_zip_and_folder_equivalent_and_alteration_rejected():
    original, metadata = read_evidence(EVIDENCE)
    assert original["replay_ids"].shape == (140000,)
    assert metadata["protocol"]["replay"]["rows"] == 140000
    with TemporaryDirectory() as temp:
        folder = Path(temp) / "task2_discrimination_study"
        with ZipFile(EVIDENCE) as archive:
            for suffix in EVIDENCE_SHA:
                member = next(name for name in archive.namelist()
                              if name.endswith("task2_discrimination_study/" + suffix))
                target = folder / suffix
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))
        extracted, _ = read_evidence(folder)
        assert np.array_equal(original["replay_ids"], extracted["replay_ids"])
        assert original["transformer_validation"] == extracted["transformer_validation"]
        (folder / "findings.json").write_text("{}")
        with pytest.raises(ValueError, match="evidence changed"):
            read_evidence(folder)


def test_summary_keeps_each_new_class_separate():
    matrix = np.eye(8, dtype=int) * 10
    matrix[5, 5] = 0
    matrix[5, 0] = 10
    metrics = {"confusion_matrix": matrix.tolist(), "accuracy": 0.875,
               "macro_f1": 0.7, "benign_false_positive": {"rate": 0.0},
               "per_class": {str(i): {"recall": float(matrix[i, i] / matrix[i].sum())}
                             for i in range(8)}}
    result = summarize(metrics)
    assert result["new_recall_each"] == {"5": 0.0, "6": 1.0, "7": 1.0}
    assert result["new_accuracy"] == pytest.approx(2 / 3)
