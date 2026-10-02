"""Deterministic, memory-bounded class-balanced sampling utilities."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .loader import iter_dataset_batches


BROAD_CLASS_ORDER = (
    "Benign",
    "DDoS",
    "DoS",
    "Botnet",
    "Bruteforce",
    "Infiltration",
    "Webattack",
    "Portscan",
)
"""Stable broad-class order for Review-2 sampling and integer label encoding."""


class SamplingError(ValueError):
    """Raised when a requested deterministic sample cannot be built."""


@dataclass(frozen=True)
class ClassBalancedSample:
    """Selected raw-file row indices and their broad labels.

    ``row_indices`` identify rows in the original Parquet file. No feature values
    are materialized or written while selecting this sample.
    """

    row_indices: np.ndarray
    labels: np.ndarray
    class_order: tuple[str, ...]
    available_counts: dict[str, int]
    per_class_limit: int
    seed: int


def _validate_requested_classes(classes: Sequence[str]) -> tuple[str, ...]:
    class_order = tuple(classes)
    if not class_order:
        raise SamplingError("At least one class must be requested.")
    if len(set(class_order)) != len(class_order):
        raise SamplingError("Requested classes must be unique.")
    return class_order


def sample_class_balanced_indices(
    path: str | Path,
    *,
    per_class_limit: int,
    seed: int,
    classes: Sequence[str] = BROAD_CLASS_ORDER,
    batch_size: int = 65_536,
    strict_schema: bool = True,
) -> ClassBalancedSample:
    """Select equal-sized, seeded class samples with reservoir sampling.

    The function scans only ``ClassLabel``. Each class reservoir stores at most
    ``per_class_limit`` original row indices, so memory use does not grow with
    dataset size. For a fixed file, class order, limit, and seed, results are
    deterministic and every encountered row within a class has equal selection
    probability.
    """

    if per_class_limit <= 0:
        raise SamplingError("per_class_limit must be a positive integer.")
    if batch_size <= 0:
        raise SamplingError("batch_size must be a positive integer.")

    class_order = _validate_requested_classes(classes)
    requested = set(class_order)
    rng = np.random.default_rng(seed)
    seen_counts: Counter[str] = Counter()
    reservoirs: dict[str, list[int]] = {name: [] for name in class_order}
    row_offset = 0

    for batch in iter_dataset_batches(
        path,
        batch_size=batch_size,
        strict_schema=strict_schema,
        columns=("ClassLabel",),
    ):
        labels = batch.column(0).to_pylist()
        for local_index, label in enumerate(labels):
            if label not in requested:
                continue

            seen_counts[label] += 1
            reservoir = reservoirs[label]
            if len(reservoir) < per_class_limit:
                reservoir.append(row_offset + local_index)
                continue

            replacement_index = int(rng.integers(seen_counts[label]))
            if replacement_index < per_class_limit:
                reservoir[replacement_index] = row_offset + local_index

        row_offset += batch.num_rows

    insufficient = [
        f"{name} ({seen_counts[name]} available)"
        for name in class_order
        if seen_counts[name] < per_class_limit
    ]
    if insufficient:
        raise SamplingError(
            "Requested per-class limit cannot be satisfied: " + ", ".join(insufficient)
        )

    row_indices = np.concatenate(
        [np.asarray(reservoirs[name], dtype=np.int64) for name in class_order]
    )
    labels = np.concatenate(
        [np.full(per_class_limit, name, dtype=object) for name in class_order]
    )

    return ClassBalancedSample(
        row_indices=row_indices,
        labels=labels,
        class_order=class_order,
        available_counts={name: seen_counts[name] for name in class_order},
        per_class_limit=per_class_limit,
        seed=seed,
    )
