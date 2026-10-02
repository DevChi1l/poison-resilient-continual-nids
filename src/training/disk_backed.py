"""Bounded-batch training over row-aligned .npy stores, without full X copies."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from src.data.large_data import TRAIN, VALIDATION, TEST, large_data_paths
from src.data.preprocessing import FittedPreprocessor


class DiskBackedFlowDataset:
    """A partition/class view of one raw float32 memmap and frozen transform.

    ``row_ids`` can restrict a benchmark to a deterministic training-only
    subset. It stores only integer positions, not copied feature rows.
    """

    def __init__(
        self,
        root: str | Path,
        preprocessor: FittedPreprocessor,
        *,
        partition: str,
        class_ids: Sequence[int],
        row_ids: np.ndarray | None = None,
    ) -> None:
        paths = large_data_paths(root)
        self.features = np.load(paths.raw_features, mmap_mode="r", allow_pickle=False)
        self.labels = np.load(paths.labels, mmap_mode="r", allow_pickle=False)
        self.splits = np.load(paths.splits, mmap_mode="r", allow_pickle=False)
        self.original_ids = np.load(paths.row_ids, mmap_mode="r", allow_pickle=False)
        if self.features.ndim != 2 or self.features.dtype != np.float32:
            raise ValueError("Expected a two-dimensional float32 raw-feature store")
        n = self.features.shape[0]
        if any(array.shape != (n,) for array in (self.labels, self.splits, self.original_ids)):
            raise ValueError("Disk-backed labels, splits, and row IDs must align")
        if tuple(preprocessor.feature_names) != tuple(
            json.loads((paths.root / "features_manifest.json").read_text())["feature_names"]
        ):
            raise ValueError("Preprocessor feature order differs from disk feature order")
        split_code = {"train": TRAIN, "validation": VALIDATION, "test": TEST}.get(partition)
        if split_code is None:
            raise ValueError("partition must be train, validation, or test")
        ids = np.asarray(class_ids, dtype=np.int64)
        if ids.ndim != 1 or ids.size == 0 or np.unique(ids).size != ids.size or np.any((ids < 0) | (ids > 7)):
            raise ValueError("class_ids must be unique global IDs in 0..7")
        if row_ids is None:
            indices = np.flatnonzero((self.splits == split_code) & np.isin(self.labels, ids))
        else:
            indices = np.asarray(row_ids, dtype=np.int64)
            if indices.ndim != 1 or len(np.unique(indices)) != len(indices) or np.any((indices < 0) | (indices >= n)):
                raise ValueError("row_ids must be unique valid original positions")
            if np.any(self.splits[indices] != split_code) or not np.isin(self.labels[indices], ids).all():
                raise ValueError("row_ids cross partition or class boundaries")
            indices = indices.copy()
        if indices.size == 0:
            raise ValueError("Disk-backed view has no rows")
        self.indices = indices
        self.class_ids = tuple(int(value) for value in ids)
        self.partition = partition
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.indices)

    def batch(self, original_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = np.asarray(original_ids, dtype=np.int64)
        if rows.ndim != 1 or rows.size == 0:
            raise ValueError("batch row IDs must be a nonempty vector")
        raw = self.features[rows]
        transformed = self.preprocessor.transform(raw)
        return transformed, self.labels[rows].astype(np.int64, copy=True), self.original_ids[rows].copy()

    def iter_epoch(
        self,
        *,
        batch_size: int,
        seed: int,
        epoch: int,
        mode: str = "shuffled",
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Seeded order; balanced mode samples class buckets with replacement.

        Class-bucket integer indices use O(n) metadata but no per-row float64
        weight vector. Epoch is part of the seed, so epochs do not repeat.
        """

        if batch_size < 1 or seed < 0 or epoch < 0:
            raise ValueError("batch_size, seed, and epoch must be valid nonnegative integers")
        rng = np.random.default_rng(np.random.SeedSequence((seed, epoch)))
        if mode == "shuffled":
            order = rng.permutation(self.indices)
            for start in range(0, len(order), batch_size):
                yield self.batch(order[start:start + batch_size])
        elif mode == "class_balanced":
            present, counts = np.unique(self.labels[self.indices], return_counts=True)
            buckets = [self.indices[self.labels[self.indices] == class_id] for class_id in present]
            for start in range(0, len(self.indices), batch_size):
                size = min(batch_size, len(self.indices) - start)
                chosen_classes = rng.integers(0, len(present), size=size)
                chosen = np.empty(size, dtype=np.int64)
                for bucket_id, bucket in enumerate(buckets):
                    where = np.flatnonzero(chosen_classes == bucket_id)
                    if where.size:
                        chosen[where] = bucket[rng.integers(0, len(bucket), size=where.size)]
                yield self.batch(chosen)
        else:
            raise ValueError("mode must be shuffled or class_balanced")


def fit_disk_backed(model, train: DiskBackedFlowDataset,
                    validation: DiskBackedFlowDataset | None = None,
                    *, max_epochs: int | None = None, verbose: bool = False):
    """Fit the existing classifier with bounded batches and best-val restoration.

    Uses the model's network, config, class mapping, save/load format, AdamW,
    cross entropy, and validation-loss early stopping. With no validation,
    this can benchmark training-only updates but does not select a checkpoint.
    """

    import torch
    from src.models.tabular_transformer import TrainingHistory

    if train.partition != "train":
        raise ValueError("Training view must contain only training partition rows")
    if validation is not None and validation.partition != "validation":
        raise ValueError("Validation view must contain only validation partition rows")
    if train.class_ids != model.class_ids:
        raise ValueError("Model class_ids must match disk training view")
    if validation is not None and validation.class_ids != model.class_ids:
        raise ValueError("Validation class_ids must match model")
    if train.features.shape[1] != model.config.num_features:
        raise ValueError("Feature width does not match model")
    epochs = model.config.epochs if max_epochs is None else max_epochs
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("max_epochs must be a positive integer")
    optimizer = torch.optim.AdamW(model.network.parameters(), lr=model.config.learning_rate,
                                  weight_decay=model.config.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    use_amp = model.config.mixed_precision and model.device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    seed = model.config.seed if model.config.sampler_seed is None else model.config.sampler_seed
    history = TrainingHistory([], [], [], 0, None, training_sampler=model.config.training_sampler,
                              sampler_seed=seed)
    counts = Counter(int(label) for label in train.labels[train.indices])
    if model.config.training_sampler == "class_balanced":
        history.expected_class_probabilities = {key: 1 / len(counts) for key in counts}
    else:
        history.expected_class_probabilities = {key: value / len(train) for key, value in counts.items()}
    history.sampled_counts_by_class = {key: 0 for key in model.class_ids}
    best_loss = float("inf")
    best_state = None
    stale = 0
    class_to_column = {value: column for column, value in enumerate(model.class_ids)}
    for epoch in range(epochs):
        model.network.train()
        loss_sum = 0.0
        epoch_counts = {key: 0 for key in model.class_ids}
        for x, labels, _ in train.iter_epoch(batch_size=model.config.batch_size,
                                             seed=seed, epoch=epoch,
                                             mode=model.config.training_sampler):
            encoded = np.fromiter((class_to_column[int(y)] for y in labels),
                                  dtype=np.int64, count=len(labels))
            for class_id, count in zip(*np.unique(labels, return_counts=True)):
                epoch_counts[int(class_id)] += int(count)
                history.sampled_counts_by_class[int(class_id)] += int(count)
            inputs = torch.as_tensor(x, dtype=torch.float32, device=model.device)
            targets = torch.as_tensor(encoded, dtype=torch.long, device=model.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = criterion(model.network(inputs), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach().item()) * len(labels)
            history.training_draws += len(labels)
            history.optimizer_steps += 1
        history.epochs_completed = epoch + 1
        history.sampled_counts_by_epoch.append(epoch_counts)
        history.train_loss.append(loss_sum / len(train))
        if validation is None:
            if verbose:
                print(f"Epoch {epoch + 1}: train_loss={history.train_loss[-1]:.6f}", flush=True)
            continue
        model.network.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for x, labels, _ in validation.iter_epoch(batch_size=model.config.batch_size,
                                                       seed=seed, epoch=0, mode="shuffled"):
                encoded = np.fromiter((class_to_column[int(y)] for y in labels),
                                      dtype=np.int64, count=len(labels))
                inputs = torch.as_tensor(x, dtype=torch.float32, device=model.device)
                targets = torch.as_tensor(encoded, dtype=torch.long, device=model.device)
                logits = model.network(inputs)
                val_loss += float(criterion(logits, targets).item()) * len(labels)
                val_correct += int((logits.argmax(dim=1) == targets).sum().item())
        val_loss /= len(validation)
        history.validation_loss.append(val_loss)
        history.validation_accuracy.append(val_correct / len(validation))
        if verbose:
            print(f"Epoch {epoch + 1}: train_loss={history.train_loss[-1]:.6f} "
                  f"val_loss={val_loss:.6f} val_accuracy={history.validation_accuracy[-1]:.4f}",
                  flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.network.state_dict().items()}
            history.best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if model.config.early_stopping_patience is not None and stale >= model.config.early_stopping_patience:
                break
    if best_state is not None:
        model.network.load_state_dict(best_state)
    return history
