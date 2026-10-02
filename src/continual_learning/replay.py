"""Deterministic class-balanced exemplars from Task 1 training only."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np

from src.data import PreparedPartition


@dataclass(frozen=True)
class ReplaySelection:
    features: np.ndarray
    labels: np.ndarray
    row_indices: np.ndarray
    source_train_positions: np.ndarray
    class_counts: dict[int, int]
    seed: int
    per_class: int

    def as_fit_replay(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the existing classifier ``fit(..., replay=(X, y))`` contract."""

        return self.features.copy(), self.labels.copy()


def select_balanced_replay(
    task1_train: PreparedPartition,
    *,
    class_ids: Sequence[int],
    per_class: int = 100,
    seed: int = 42,
    forbidden_row_indices: Sequence[int] = (),
) -> ReplaySelection:
    """Select unique Task 1 training exemplars per old class without replacement.

    ``forbidden_row_indices`` should combine validation and test raw row IDs.
    The helper rejects overlap, returning copies so the stored buffer cannot
    mutate its source partition.
    """

    X = np.asarray(task1_train.features)
    y = np.asarray(task1_train.labels)
    rows = np.asarray(task1_train.row_indices)
    ids = np.asarray(class_ids)
    forbidden = np.asarray(forbidden_row_indices)
    if (X.ndim != 2 or y.ndim != 1 or rows.ndim != 1 or X.shape[0] != y.size or
            y.size != rows.size or y.size == 0 or not np.issubdtype(y.dtype, np.integer) or
            not np.issubdtype(rows.dtype, np.integer) or np.unique(rows).size != rows.size):
        raise ValueError("Task 1 training partition must have aligned unique rows")
    if (ids.ndim != 1 or ids.size == 0 or not np.issubdtype(ids.dtype, np.integer) or
            np.unique(ids).size != ids.size or not np.isin(y, ids).all()):
        raise ValueError("class_ids must be unique and cover Task 1 labels")
    if not isinstance(per_class, Integral) or isinstance(per_class, (bool, np.bool_)) or per_class < 1:
        raise ValueError("per_class must be a positive integer")
    if not isinstance(seed, Integral) or isinstance(seed, (bool, np.bool_)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if forbidden.size and (forbidden.ndim != 1 or not np.issubdtype(forbidden.dtype, np.integer)):
        raise ValueError("forbidden_row_indices must be integer row IDs")
    if np.isin(rows, forbidden).any():
        raise ValueError("Task 1 train overlaps forbidden validation/test rows")
    rng = np.random.default_rng(int(seed))
    selected = []
    for class_id in ids:
        candidates = np.flatnonzero(y == class_id)
        if candidates.size < per_class:
            raise ValueError(f"class {int(class_id)} has only {candidates.size} training rows")
        selected.append(np.sort(rng.choice(candidates, size=int(per_class), replace=False)))
    positions = np.concatenate(selected).astype(np.int64)
    return ReplaySelection(
        features=X[positions].copy(),
        labels=y[positions].copy(),
        row_indices=rows[positions].copy(),
        source_train_positions=positions.copy(),
        class_counts={int(class_id): int(per_class) for class_id in ids},
        seed=int(seed),
        per_class=int(per_class),
    )
