"""Seeded, stratified train/validation/test split utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class SplitError(ValueError):
    """Raised when labels cannot support the requested stratified split."""


@dataclass(frozen=True)
class StratifiedSplit:
    """Positions into a sampled dataset for reproducible train/validation/test use."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    seed: int


def _allocate_class_counts(
    class_size: int,
    fractions: tuple[float, float, float],
) -> tuple[int, int, int]:
    exact = np.asarray(fractions, dtype=float) * class_size
    allocated = np.floor(exact).astype(int)
    remainder = class_size - int(allocated.sum())
    for index in np.argsort(-(exact - allocated), kind="stable")[:remainder]:
        allocated[index] += 1
    return tuple(int(value) for value in allocated)


def stratified_split_indices(
    labels: np.ndarray,
    *,
    seed: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> StratifiedSplit:
    """Split each class separately, preserving class balance in all partitions."""

    fractions = (train_fraction, validation_fraction, test_fraction)
    if not np.isclose(sum(fractions), 1.0):
        raise SplitError("Train, validation, and test fractions must sum to 1.0.")
    if any(fraction <= 0 for fraction in fractions):
        raise SplitError("All split fractions must be positive.")

    values = np.asarray(labels)
    if values.ndim != 1 or values.size == 0:
        raise SplitError("labels must be a non-empty one-dimensional array.")

    rng = np.random.default_rng(seed)
    partitions: list[list[np.ndarray]] = [[], [], []]
    for label in np.unique(values):
        class_positions = np.flatnonzero(values == label)
        if class_positions.size < 3:
            raise SplitError(
                f"Class {label!r} has only {class_positions.size} rows; at least 3 are required."
            )

        train_count, validation_count, test_count = _allocate_class_counts(
            class_positions.size,
            fractions,
        )
        if min(train_count, validation_count, test_count) == 0:
            raise SplitError(
                f"Class {label!r} cannot populate every split with the requested fractions."
            )

        shuffled = rng.permutation(class_positions)
        train_end = train_count
        validation_end = train_end + validation_count
        partitions[0].append(shuffled[:train_end])
        partitions[1].append(shuffled[train_end:validation_end])
        partitions[2].append(shuffled[validation_end:validation_end + test_count])

    return StratifiedSplit(
        train=np.concatenate(partitions[0]),
        validation=np.concatenate(partitions[1]),
        test=np.concatenate(partitions[2]),
        seed=seed,
    )
