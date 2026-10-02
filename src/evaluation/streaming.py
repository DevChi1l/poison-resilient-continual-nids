"""Confusion-based classification metrics accumulated from bounded batches."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


class ConfusionAccumulator:
    """Accumulate external class IDs without retaining per-row predictions."""

    def __init__(self, class_ids: Sequence[int]) -> None:
        ids = np.asarray(class_ids)
        if (ids.ndim != 1 or ids.size == 0 or
                not np.issubdtype(ids.dtype, np.integer) or
                np.unique(ids).size != ids.size):
            raise ValueError("class_ids must be a nonempty unique integer vector")
        self.class_ids = tuple(int(value) for value in ids)
        self._column = {value: index for index, value in enumerate(self.class_ids)}
        self.counts = np.zeros((len(ids), len(ids)), dtype=np.int64)

    def update(self, truth: np.ndarray, predicted: np.ndarray) -> None:
        actual = np.asarray(truth)
        guesses = np.asarray(predicted)
        if (actual.ndim != 1 or guesses.shape != actual.shape or
                not np.issubdtype(actual.dtype, np.integer) or
                not np.issubdtype(guesses.dtype, np.integer)):
            raise ValueError("truth and predictions must be aligned integer vectors")
        if not np.isin(actual, self.class_ids).all() or not np.isin(guesses, self.class_ids).all():
            raise ValueError("unknown external class ID")
        rows = np.fromiter((self._column[int(value)] for value in actual),
                           dtype=np.int64, count=len(actual))
        cols = np.fromiter((self._column[int(value)] for value in guesses),
                           dtype=np.int64, count=len(guesses))
        np.add.at(self.counts, (rows, cols), 1)

    def metrics(self, *, class_names: Mapping[int, str] | None = None,
                benign_class_id: int | None = None) -> dict:
        if self.counts.sum() == 0:
            raise ValueError("no evaluation rows accumulated")
        per_class = {}
        recalls = []
        f1s = []
        for column, class_id in enumerate(self.class_ids):
            tp = int(self.counts[column, column])
            support = int(self.counts[column].sum())
            predicted = int(self.counts[:, column].sum())
            precision = tp / predicted if predicted else 0.0
            recall = tp / support if support else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            per_class[str(class_id)] = {
                "name": class_names[class_id] if class_names is not None else str(class_id),
                "precision": precision, "recall": recall, "f1": f1, "support": support,
            }
            recalls.append(recall)
            f1s.append(f1)
        result = {
            "class_ids": list(self.class_ids),
            "total_rows": int(self.counts.sum()),
            "accuracy": float(np.trace(self.counts) / self.counts.sum()),
            "macro_f1": float(np.mean(f1s)),
            "balanced_accuracy": float(np.mean(recalls)),
            "per_class": per_class,
            "confusion_matrix": self.counts.tolist(),
        }
        if benign_class_id is not None:
            if benign_class_id not in self._column:
                raise ValueError("benign_class_id must belong to class_ids")
            row = self.counts[self._column[benign_class_id]]
            support = int(row.sum())
            errors = support - int(row[self._column[benign_class_id]])
            result["benign_false_positive"] = {
                "class_id": int(benign_class_id), "count": errors,
                "support": support, "rate": errors / support if support else None,
            }
        return result


def evaluate_disk_backed(model, dataset, *, batch_size: int,
                         class_names: Mapping[int, str] | None = None,
                         benign_class_id: int | None = None) -> dict:
    """Run bounded inference once and retain only confusion counts."""

    import torch

    if batch_size < 1 or dataset.class_ids != model.class_ids:
        raise ValueError("batch size or class IDs do not match the model")
    accumulator = ConfusionAccumulator(model.class_ids)
    model.network.eval()
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            x, truth, _ = dataset.batch(dataset.indices[start:start + batch_size])
            inputs = torch.as_tensor(x, dtype=torch.float32, device=model.device)
            columns = model.network(inputs).argmax(dim=1).cpu().numpy()
            predictions = np.asarray(model.class_ids, dtype=np.int64)[columns]
            accumulator.update(truth, predictions)
    return accumulator.metrics(class_names=class_names, benign_class_id=benign_class_id)
