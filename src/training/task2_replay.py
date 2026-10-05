"""Bounded Task 2 plus frozen old-replay batches for full-row training."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


class Task2ReplayView:
    """Draw over present supplied classes, then within each class.

    New Task 2 rows are read and transformed by the disk view on demand.
    Replay features are already frozen, transformed float32 rows and are
    copied directly into batches; they must never be preprocessed again.
    No full feature matrix or per-row floating sampling weights are built.

    The default remains uniform stochastic class sampling. An optional class
    probability plan is converted once into deterministic integer quotas with
    largest-remainder allocation, so an epoch records the requested exposure
    exactly while retaining within-class sampling with replacement.
    """

    def __init__(self, task2_view, replay_features: np.ndarray,
                 replay_labels: np.ndarray, replay_original_ids: np.ndarray,
                 *, class_ids: Sequence[int] = tuple(range(8)),
                 draws_per_epoch: int = 70_576,
                 class_draw_probabilities: Mapping[int, float] | None = None) -> None:
        ids = np.asarray(class_ids)
        X = np.asarray(replay_features)
        y = np.asarray(replay_labels)
        raw_ids = np.asarray(replay_original_ids)
        if (task2_view.partition != "train" or
                tuple(task2_view.class_ids) != (5, 6, 7) or
                ids.ndim != 1 or ids.size != 8 or
                not np.array_equal(ids, np.arange(8)) or
                X.ndim != 2 or X.dtype != np.float32 or
                X.shape[1] != task2_view.features.shape[1] or
                not np.isfinite(X).all() or
                y.shape != (len(X),) or not np.issubdtype(y.dtype, np.integer) or
                not np.isin(y, np.arange(5)).all() or
                raw_ids.shape != (len(X),) or
                not np.issubdtype(raw_ids.dtype, np.integer) or
                len(np.unique(raw_ids)) != len(raw_ids) or
                np.any((raw_ids < 0) | (raw_ids >= len(task2_view.splits))) or
                np.any(task2_view.splits[raw_ids] != 0) or
                not np.isin(task2_view.labels[raw_ids], np.arange(5)).all() or
                isinstance(draws_per_epoch, bool) or
                not isinstance(draws_per_epoch, int) or draws_per_epoch < 1):
            raise ValueError("Expected full Task 2 training view and valid frozen old replay")
        self.task2_view = task2_view
        self.features = task2_view.features  # width contract for durable trainer
        self.partition = "train"
        self.class_ids = tuple(int(value) for value in ids)
        self.replay_features = X.copy()
        self.replay_labels = y.astype(np.int64, copy=True)
        self.replay_original_ids = raw_ids.astype(np.int64, copy=True)
        self.draws_per_epoch = draws_per_epoch
        self._buckets = {}
        for class_id in self.class_ids:
            if class_id < 5:
                positions = np.flatnonzero(self.replay_labels == class_id)
            else:
                positions = task2_view.indices[task2_view.labels[task2_view.indices] == class_id]
            if positions.size:
                self._buckets[class_id] = positions
        if not all(class_id in self._buckets for class_id in (5, 6, 7)):
            raise ValueError("All three new Task 2 classes must be present")
        self.class_draw_probabilities = self._validate_probabilities(
            class_draw_probabilities)
        self.planned_class_draw_counts = self._allocate_class_draws(
            self.class_draw_probabilities)
        self.last_epoch_exposure: dict | None = None

    def _validate_probabilities(
        self, values: Mapping[int, float] | None,
    ) -> dict[int, float] | None:
        if values is None:
            return None
        try:
            probabilities = {int(key): float(value) for key, value in values.items()}
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("class_draw_probabilities must map class IDs to numbers") from error
        present = set(self.present_class_ids)
        if (set(probabilities) != present or
                any(not np.isfinite(value) or value <= 0
                    for value in probabilities.values()) or
                not np.isclose(sum(probabilities.values()), 1.0,
                               rtol=0.0, atol=1e-12)):
            raise ValueError(
                "class_draw_probabilities must contain each present class exactly once "
                "with positive finite values summing to one"
            )
        return {class_id: probabilities[class_id] for class_id in self.present_class_ids}

    def _allocate_class_draws(
        self, probabilities: Mapping[int, float] | None,
    ) -> dict[int, int] | None:
        if probabilities is None:
            return None
        present = self.present_class_ids
        raw = np.asarray([probabilities[class_id] * self.draws_per_epoch
                          for class_id in present], dtype=np.float64)
        allocated = np.floor(raw).astype(np.int64)
        remaining = self.draws_per_epoch - int(allocated.sum())
        # Stable class-ID tie breaking keeps the quota calculation reproducible.
        order = sorted(range(len(present)), key=lambda index: (-float(raw[index] % 1),
                                                                present[index]))
        for index in order[:remaining]:
            allocated[index] += 1
        if np.any(allocated < 1) or int(allocated.sum()) != self.draws_per_epoch:
            raise ValueError("Every present class must receive at least one epoch draw")
        return {class_id: int(allocated[index])
                for index, class_id in enumerate(present)}

    def __len__(self) -> int:
        return self.draws_per_epoch

    @property
    def present_class_ids(self) -> tuple[int, ...]:
        return tuple(self._buckets)

    def iter_epoch(self, *, batch_size: int, seed: int, epoch: int,
                   mode: str = "class_balanced"):
        if (mode != "class_balanced" or batch_size < 1 or
                isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or
                isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0):
            raise ValueError("Require class-balanced mode, positive batch, seed, and epoch")
        rng = np.random.default_rng(np.random.SeedSequence((seed, epoch)))
        present = self.present_class_ids
        counts = {str(class_id): 0 for class_id in self.class_ids}
        unique_replay: set[int] = set()
        unique_rows = {class_id: set() for class_id in present}
        planned_classes = None
        if self.planned_class_draw_counts is not None:
            planned_classes = np.concatenate([
                np.full(self.planned_class_draw_counts[class_id], bucket_number,
                        dtype=np.int64)
                for bucket_number, class_id in enumerate(present)
            ])
            rng.shuffle(planned_classes)
        for start in range(0, len(self), batch_size):
            size = min(batch_size, len(self) - start)
            chosen_classes = (rng.integers(0, len(present), size=size)
                              if planned_classes is None
                              else planned_classes[start:start + size])
            X = np.empty((size, self.features.shape[1]), dtype=np.float32)
            y = np.empty(size, dtype=np.int64)
            row_ids = np.empty(size, dtype=np.int64)
            for bucket_number, class_id in enumerate(present):
                at = np.flatnonzero(chosen_classes == bucket_number)
                if not len(at):
                    continue
                bucket = self._buckets[class_id]
                choices = bucket[rng.integers(0, len(bucket), size=len(at))]
                if class_id < 5:
                    X[at] = self.replay_features[choices]
                    y[at] = self.replay_labels[choices]
                    row_ids[at] = self.replay_original_ids[choices]
                    unique_replay.update(int(value) for value in choices)
                else:
                    fresh_x, fresh_y, fresh_ids = self.task2_view.batch(choices)
                    X[at], y[at], row_ids[at] = fresh_x, fresh_y, fresh_ids
                unique_rows[class_id].update(int(value) for value in row_ids[at])
                counts[str(class_id)] += len(at)
            yield X, y, row_ids
        self.last_epoch_exposure = {
            "epoch": epoch + 1, "sampler_seed": seed,
            "mode": ("class_balanced_present_classes_with_replacement"
                     if self.planned_class_draw_counts is None
                     else "fixed_class_quota_with_replacement"),
            "expected_class_probabilities": {
                str(class_id): (1 / len(present) if self.class_draw_probabilities is None
                                else self.class_draw_probabilities[class_id])
                for class_id in present
            },
            "expected_probability_per_present_class": (
                1 / len(present) if self.class_draw_probabilities is None else None
            ),
            "planned_class_draw_counts": (
                None if self.planned_class_draw_counts is None else
                {str(key): value for key, value in self.planned_class_draw_counts.items()}
            ),
            "present_class_ids": list(present),
            "sampled_counts": counts,
            "unique_rows_drawn_by_class": {
                str(class_id): len(unique_rows[class_id]) for class_id in present
            },
            "draws": len(self),
            "stored_replay_rows": len(self.replay_labels),
            "unique_replay_rows_drawn": len(unique_replay),
        }
