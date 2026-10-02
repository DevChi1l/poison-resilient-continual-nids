"""Build two class-incremental tasks with Task-1-only preprocessing fit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.data import (
    FittedPreprocessor,
    PreparedPartition,
    SelectedRows,
    StratifiedSplit,
    encode_broad_labels,
    fit_preprocessor,
)
from src.data.sampling import BROAD_CLASS_ORDER


TASK1_CLASS_IDS = (0, 1, 2, 3, 4)
TASK2_CLASS_IDS = (5, 6, 7)


@dataclass(frozen=True)
class TaskPartitions:
    class_ids: tuple[int, ...]
    train: PreparedPartition
    validation: PreparedPartition
    test: PreparedPartition


@dataclass(frozen=True)
class ContinualTasks:
    task1: TaskPartitions
    task2: TaskPartitions
    seen_validation: PreparedPartition
    combined_test: PreparedPartition
    class_names: dict[int, str]
    feature_names: tuple[str, ...]
    preprocessor: FittedPreprocessor
    split: StratifiedSplit


def prepare_two_task_dataset(
    selected: SelectedRows,
    split: StratifiedSplit,
    class_order: Sequence[str] = BROAD_CLASS_ORDER,
) -> ContinualTasks:
    """Preserve global IDs and fit preprocessing on Task 1 training rows only.

    ``selected`` should come from ``materialize_selected_rows`` in the same
    order as the seeded sample. The existing split positions cover that sample.
    No validation, test, or future Task 2 row is used to fit statistics.
    """

    raw_features = np.asarray(selected.features)
    raw_rows = np.asarray(selected.row_indices)
    raw_labels = np.asarray(selected.labels)
    n_rows = len(raw_rows)
    if (raw_features.ndim != 2 or raw_features.shape[0] != n_rows or
            raw_labels.shape != (n_rows,) or n_rows == 0 or
            raw_rows.ndim != 1 or not np.issubdtype(raw_rows.dtype, np.integer) or
            np.any(raw_rows < 0) or np.unique(raw_rows).size != n_rows):
        raise ValueError("selected rows, labels, and features must be aligned and unique")
    positions = [np.asarray(getattr(split, name)) for name in ("train", "validation", "test")]
    if any(p.ndim != 1 or not np.issubdtype(p.dtype, np.integer) for p in positions):
        raise ValueError("split positions must be one-dimensional integer arrays")
    joined = np.concatenate(positions)
    if joined.size != n_rows or np.unique(joined).size != n_rows or joined.min() < 0 or joined.max() >= n_rows:
        raise ValueError("split positions must cover selected rows exactly once")
    if len(class_order) != 8 or tuple(class_order) != BROAD_CLASS_ORDER:
        raise ValueError("the Review-2 protocol requires the canonical eight-class order")
    encoded, class_names = encode_broad_labels(raw_labels, class_order)
    if set(np.unique(encoded)) != set(TASK1_CLASS_IDS + TASK2_CLASS_IDS):
        raise ValueError("all eight global broad classes must be present")
    train_task1_positions = split.train[np.isin(encoded[split.train], TASK1_CLASS_IDS)]
    if train_task1_positions.size == 0:
        raise ValueError("Task 1 training rows are required")
    preprocessor = fit_preprocessor(raw_features[train_task1_positions], selected.feature_names)

    def partition(indices: np.ndarray) -> PreparedPartition:
        return PreparedPartition(
            features=preprocessor.transform(raw_features[indices]),
            labels=encoded[indices].copy(),
            row_indices=raw_rows[indices].copy(),
        )

    def task(class_ids: tuple[int, ...]) -> TaskPartitions:
        parts = {
            name: partition(p[np.isin(encoded[p], class_ids)])
            for name, p in zip(("train", "validation", "test"), positions)
        }
        if any(part.labels.size == 0 or set(np.unique(part.labels)) != set(class_ids) for part in parts.values()):
            raise ValueError("every task partition must contain all its global class IDs")
        return TaskPartitions(class_ids, **parts)

    return ContinualTasks(
        task1=task(TASK1_CLASS_IDS),
        task2=task(TASK2_CLASS_IDS),
        seen_validation=partition(split.validation),
        combined_test=partition(split.test),
        class_names=class_names,
        feature_names=tuple(selected.feature_names),
        preprocessor=preprocessor,
        split=split,
    )
