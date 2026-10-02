"""Seeded random and targeted label flips for training labels only."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite
from numbers import Integral, Real
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PoisoningResult:
    """Copied labels, changed positions, and explicit attack-budget accounting."""

    original_labels: np.ndarray
    poisoned_labels: np.ndarray
    changed_indices: np.ndarray
    changed_original_row_indices: np.ndarray | None
    mode: str
    requested_rate: float
    achieved_rate: float
    seed: int
    total_count: int
    eligible_count: int
    changed_count: int
    achieved_eligible_fraction: float
    achieved_total_fraction: float
    budget_denominator: str
    allowed_class_ids: tuple[int, ...]
    source_class_ids: tuple[int, ...] | None
    target_class_id: int | None

    @property
    def changed_labels(self) -> np.ndarray:
        """Compatibility alias for the documented changed-label result."""

        return self.poisoned_labels


def _integer_ids(values: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    return array


def apply_label_flip(
    labels: np.ndarray,
    rate: float,
    seed: int,
    allowed_classes: Sequence[int] | None = None,
    *,
    mode: str = "random",
    source_class_ids: Sequence[int] | None = None,
    target_class_id: int | None = None,
    allowed_class_ids: Sequence[int] | None = None,
    original_row_indices: np.ndarray | None = None,
) -> PoisoningResult:
    """Flip a rounded fraction of the selected training-label budget.

    The budget is ``len(labels)`` in random mode and the number of eligible
    source rows in targeted mode. The flip count uses half-up rounding:
    ``floor(rate * budget + 0.5)``. Selection is without replacement.
    ``allowed_classes`` is the original API name; use ``allowed_class_ids``
    for new callers. Neither labels nor original row indices are mutated.
    """

    original = _integer_ids(labels, "labels").copy()
    if isinstance(rate, (bool, np.bool_)) or not isinstance(rate, Real):
        raise ValueError("rate must be a finite real scalar in [0, 1]")
    requested_rate = float(rate)
    if not isfinite(requested_rate) or not 0.0 <= requested_rate <= 1.0:
        raise ValueError("rate must be a finite real scalar in [0, 1]")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if mode not in ("random", "targeted"):
        raise ValueError("mode must be 'random' or 'targeted'")
    if allowed_classes is not None and allowed_class_ids is not None:
        raise ValueError("supply only one of allowed_classes and allowed_class_ids")

    supplied_ids = allowed_class_ids if allowed_class_ids is not None else allowed_classes
    choices = (
        np.unique(original)
        if supplied_ids is None
        else _integer_ids(supplied_ids, "allowed_class_ids")
    )
    if choices.size == 0 and original.size > 0:
        raise ValueError("allowed_class_ids must contain the training classes")
    if np.unique(choices).size != choices.size:
        raise ValueError("allowed_class_ids must be unique")
    if not np.isin(original, choices).all():
        raise ValueError("every training label must be in allowed_class_ids")
    normalized_choices = tuple(int(value) for value in choices)

    raw_indices = None
    if original_row_indices is not None:
        raw_indices = _integer_ids(original_row_indices, "original_row_indices")
        if raw_indices.size != original.size or (raw_indices < 0).any():
            raise ValueError("original_row_indices must match labels and be non-negative")
        if np.unique(raw_indices).size != raw_indices.size:
            raise ValueError("original_row_indices must be unique")

    normalized_sources: tuple[int, ...] | None = None
    normalized_target: int | None = None
    if mode == "random":
        if source_class_ids is not None or target_class_id is not None:
            raise ValueError("source_class_ids and target_class_id are targeted-only")
        eligible_indices = np.arange(original.size, dtype=np.int64)
        denominator = "all_training_rows"
    else:
        if source_class_ids is None or target_class_id is None:
            raise ValueError("targeted mode requires source_class_ids and target_class_id")
        sources = _integer_ids(source_class_ids, "source_class_ids")
        if sources.size == 0 or np.unique(sources).size != sources.size:
            raise ValueError("source_class_ids must be nonempty and unique")
        if isinstance(target_class_id, (bool, np.bool_)) or not isinstance(target_class_id, Integral):
            raise ValueError("target_class_id must be an integer")
        if not np.isin(sources, choices).all() or target_class_id not in normalized_choices:
            raise ValueError("target and source class IDs must be in allowed_class_ids")
        normalized_sources = tuple(int(value) for value in sources)
        normalized_target = int(target_class_id)
        eligible_indices = np.flatnonzero(
            np.isin(original, sources) & (original != normalized_target)
        )
        denominator = "eligible_source_rows_excluding_target"

    eligible_count = int(eligible_indices.size)
    changed_count = floor(requested_rate * eligible_count + 0.5)
    if requested_rate > 0 and eligible_count == 0:
        raise ValueError("no rows are eligible for the requested poisoning mode")
    if changed_count > 0 and mode == "random" and choices.size < 2:
        raise ValueError("random flipping needs at least two allowed class IDs")

    rng = np.random.default_rng(int(seed))
    changed_indices = np.sort(
        rng.choice(eligible_indices, size=changed_count, replace=False)
    ).astype(np.int64, copy=False)
    poisoned = original.copy()
    if mode == "targeted":
        poisoned[changed_indices] = normalized_target
    else:
        for index in changed_indices:
            alternatives = choices[choices != original[index]]
            poisoned[index] = rng.choice(alternatives)

    return PoisoningResult(
        original_labels=original,
        poisoned_labels=poisoned,
        changed_indices=changed_indices,
        changed_original_row_indices=(
            raw_indices[changed_indices].copy() if raw_indices is not None else None
        ),
        mode=mode,
        requested_rate=requested_rate,
        achieved_rate=changed_count / eligible_count if eligible_count else 0.0,
        seed=int(seed),
        total_count=int(original.size),
        eligible_count=eligible_count,
        changed_count=changed_count,
        achieved_eligible_fraction=changed_count / eligible_count if eligible_count else 0.0,
        achieved_total_fraction=changed_count / original.size if original.size else 0.0,
        budget_denominator=denominator,
        allowed_class_ids=normalized_choices,
        source_class_ids=normalized_sources,
        target_class_id=normalized_target,
    )
