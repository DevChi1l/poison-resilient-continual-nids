"""Audit replay quarantine using simulator truth, separate from the gate."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def replay_gate_metrics(
    retained_mask: np.ndarray,
    changed_indices: np.ndarray,
    original_labels: np.ndarray,
    supplied_labels: np.ndarray,
    *,
    class_ids: Sequence[int],
) -> dict:
    """Evaluate rejection/retention; never feed these inputs into filtering.

    Changed-index masks and original clean labels are simulation-only audit
    information. Undefined poison rates in a clean condition are ``None``.
    """

    kept = np.asarray(retained_mask)
    changed = np.asarray(changed_indices)
    original = np.asarray(original_labels)
    supplied = np.asarray(supplied_labels)
    ids = np.asarray(class_ids)
    n = original.size
    if (original.ndim != 1 or supplied.shape != original.shape or
            not np.issubdtype(original.dtype, np.integer) or
            not np.issubdtype(supplied.dtype, np.integer) or
            kept.shape != (n,) or kept.dtype != np.bool_ or
            ids.ndim != 1 or ids.size == 0 or not np.issubdtype(ids.dtype, np.integer) or
            np.unique(ids).size != ids.size or not np.isin(original, ids).all() or
            not np.isin(supplied, ids).all()):
        raise ValueError("aligned old-class labels, boolean mask, and unique IDs required")
    if (changed.ndim != 1 or not np.issubdtype(changed.dtype, np.integer) or
            np.unique(changed).size != changed.size or np.any((changed < 0) | (changed >= n))):
        raise ValueError("changed_indices must be unique valid replay positions")
    poison = np.zeros(n, dtype=bool)
    poison[changed] = True
    if not np.array_equal(original != supplied, poison):
        raise ValueError("changed_indices must match actual altered labels")
    poisoned_count = int(poison.sum())
    clean_count = n - poisoned_count
    poison_rejected = int(np.count_nonzero(poison & ~kept))
    poison_retained = poisoned_count - poison_rejected
    clean_rejected = int(np.count_nonzero(~poison & ~kept))
    return {
        "total_candidates": n,
        "retained_total": int(kept.sum()),
        "poisoned_candidates": poisoned_count,
        "poison_rejected": poison_rejected,
        "poison_rejection_rate": poison_rejected / poisoned_count if poisoned_count else None,
        "poison_retained": poison_retained,
        "retained_poison_fraction": poison_retained / poisoned_count if poisoned_count else None,
        "clean_candidates": clean_count,
        "clean_false_rejected": clean_rejected,
        "clean_false_rejection_rate": clean_rejected / clean_count if clean_count else None,
        "retained_per_supplied_class": {
            str(int(class_id)): int(np.count_nonzero(kept & (supplied == class_id)))
            for class_id in ids
        },
    }
