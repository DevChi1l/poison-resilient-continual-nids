"""Deterministic training-only replay positions for disk-backed Task 1."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def select_disk_replay_rows(dataset, *, class_ids: Sequence[int],
                            per_class: int = 100, seed: int = 42) -> np.ndarray:
    """Return unique original-row positions, without materializing features.

    The caller can obtain frozen transformed features, labels, and original
    dataset IDs from ``dataset.batch(positions)`` and persist them separately.
    Only an explicitly training-partition disk view is accepted.
    """

    ids = np.asarray(class_ids)
    if (dataset.partition != "train" or ids.ndim != 1 or ids.size == 0 or
            not np.issubdtype(ids.dtype, np.integer) or
            np.unique(ids).size != ids.size or
            not np.isin(ids, dataset.class_ids).all() or
            isinstance(per_class, bool) or not isinstance(per_class, int) or per_class < 1 or
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError("Require training view, valid class IDs, positive count, and seed")
    positions = np.asarray(dataset.indices, dtype=np.int64)
    if positions.ndim != 1 or np.unique(positions).size != positions.size:
        raise ValueError("Disk view row positions must be unique")
    if not np.all(np.asarray(dataset.splits)[positions] == 0):
        raise ValueError("Replay candidates must belong to the training split")
    labels = np.asarray(dataset.labels)[positions]
    rng = np.random.default_rng(seed)
    chosen = []
    for class_id in ids:
        available = positions[labels == class_id]
        if len(available) < per_class:
            raise ValueError(f"Not enough training rows for class {int(class_id)}")
        chosen.append(np.sort(rng.choice(available, size=per_class, replace=False)))
    result = np.concatenate(chosen)
    if np.unique(result).size != result.size:
        raise ValueError("Replay positions are not unique")
    return result
