"""Train-only preprocessing for selected combined-flow dataset rows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .loader import inspect_parquet, iter_dataset_batches
from .sampling import ClassBalancedSample
from .splitting import StratifiedSplit


class PreprocessingError(ValueError):
    """Raised when selected rows cannot safely form a prepared dataset."""


@dataclass(frozen=True)
class SelectedRows:
    """Raw selected feature values and broad labels in caller-requested row order."""

    features: np.ndarray
    labels: np.ndarray
    row_indices: np.ndarray
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class FittedPreprocessor:
    """Training-only numerical transformation state for reproducible inference."""

    feature_names: tuple[str, ...]
    imputation_values: np.ndarray
    means: np.ndarray
    scales: np.ndarray

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Map raw feature values to imputed, standardized ``float32`` values."""

        values = _negative_or_nonfinite_to_missing(features)
        _impute_in_place(values, self.imputation_values)
        return ((values - self.means) / self.scales).astype(np.float32, copy=False)


@dataclass(frozen=True)
class PreparedPartition:
    """One transformed split ready for a model or PyTorch dataset wrapper."""

    features: np.ndarray
    labels: np.ndarray
    row_indices: np.ndarray


@dataclass(frozen=True)
class PreparedDataset:
    """Canonical Review-2 train/validation/test data contract."""

    train: PreparedPartition
    validation: PreparedPartition
    test: PreparedPartition
    feature_names: tuple[str, ...]
    class_names: dict[int, str]
    preprocessor: FittedPreprocessor


def materialize_selected_rows(
    path: str | Path,
    row_indices: Sequence[int] | np.ndarray,
    *,
    batch_size: int = 65_536,
    strict_schema: bool = True,
) -> SelectedRows:
    """Read only requested original-file rows while scanning bounded Parquet batches.

    Row indices are raw zero-based Parquet row positions. Feature rows are returned
    in exactly the supplied order, which keeps them aligned with selected labels and
    split positions. The original dataset is read only and never rewritten.
    """

    requested_indices = np.asarray(row_indices, dtype=np.int64)
    if requested_indices.ndim != 1 or requested_indices.size == 0:
        raise PreprocessingError("row_indices must be a non-empty one-dimensional array.")
    if np.any(requested_indices < 0):
        raise PreprocessingError("row_indices cannot contain negative values.")
    if len(np.unique(requested_indices)) != requested_indices.size:
        raise PreprocessingError("row_indices must be unique.")

    info = inspect_parquet(path, strict_schema=strict_schema)
    if int(requested_indices.max()) >= info.row_count:
        raise PreprocessingError(
            f"Requested row index {requested_indices.max()} is outside dataset row count "
            f"{info.row_count}."
        )

    sort_order = np.argsort(requested_indices)
    sorted_indices = requested_indices[sort_order]
    features = np.empty(
        (requested_indices.size, len(info.schema.retained_feature_names)),
        dtype=np.float64,
    )
    labels = np.empty(requested_indices.size, dtype=object)
    found_count = 0
    row_offset = 0

    for batch in iter_dataset_batches(
        path,
        batch_size=batch_size,
        strict_schema=strict_schema,
    ):
        batch_end = row_offset + batch.num_rows
        left = np.searchsorted(sorted_indices, row_offset, side="left")
        right = np.searchsorted(sorted_indices, batch_end, side="left")

        if left != right:
            local_indices = sorted_indices[left:right] - row_offset
            original_positions = sort_order[left:right]
            selected_feature_columns = [
                batch.column(index).to_numpy(zero_copy_only=False)[local_indices]
                for index in range(len(info.schema.retained_feature_names))
            ]
            features[original_positions] = np.column_stack(selected_feature_columns)
            broad_label_index = batch.schema.get_field_index("ClassLabel")
            broad_labels = batch.column(broad_label_index).to_pylist()
            labels[original_positions] = [broad_labels[index] for index in local_indices]
            found_count += right - left

        row_offset = batch_end
        if found_count == requested_indices.size:
            break

    if found_count != requested_indices.size:
        raise PreprocessingError(
            f"Loaded {found_count} requested rows, expected {requested_indices.size}."
        )

    return SelectedRows(
        features=features,
        labels=labels,
        row_indices=requested_indices.copy(),
        feature_names=info.schema.retained_feature_names,
    )


def fit_preprocessor(
    train_features: np.ndarray,
    feature_names: Sequence[str],
) -> FittedPreprocessor:
    """Fit negative-to-missing conversion, median imputation, and scaling on train rows."""

    values = _negative_or_nonfinite_to_missing(train_features)
    with np.errstate(all="ignore"):
        imputation_values = np.nanmedian(values, axis=0)
    if np.isnan(imputation_values).any():
        missing_columns = [
            name
            for name, value in zip(feature_names, imputation_values)
            if np.isnan(value)
        ]
        raise PreprocessingError(
            "A retained feature has no valid training values: " + ", ".join(missing_columns)
        )

    _impute_in_place(values, imputation_values)
    means = values.mean(axis=0)
    scales = values.std(axis=0)
    scales[scales == 0] = 1.0

    return FittedPreprocessor(
        feature_names=tuple(feature_names),
        imputation_values=imputation_values,
        means=means,
        scales=scales,
    )


def encode_broad_labels(
    labels: np.ndarray,
    class_order: Sequence[str],
) -> tuple[np.ndarray, dict[int, str]]:
    """Encode broad text labels with one stable, explicit class-to-ID mapping."""

    class_names = {index: name for index, name in enumerate(class_order)}
    label_to_id = {name: index for index, name in class_names.items()}
    unknown_labels = sorted(set(labels.tolist()) - set(label_to_id))
    if unknown_labels:
        raise PreprocessingError(
            "Selected labels are absent from the class order: " + ", ".join(unknown_labels)
        )
    return np.asarray([label_to_id[label] for label in labels], dtype=np.int64), class_names


def prepare_sampled_dataset(
    path: str | Path,
    sample: ClassBalancedSample,
    split: StratifiedSplit,
    *,
    batch_size: int = 65_536,
    strict_schema: bool = True,
) -> PreparedDataset:
    """Materialize a sample and create train-only transformed model partitions."""

    _validate_split_positions(sample.labels.size, split)
    selected = materialize_selected_rows(
        path,
        sample.row_indices,
        batch_size=batch_size,
        strict_schema=strict_schema,
    )
    if not np.array_equal(selected.labels, sample.labels):
        raise PreprocessingError(
            "Labels at sampled row indices differ from the labels recorded during sampling."
        )

    train_features = selected.features[split.train]
    preprocessor = fit_preprocessor(train_features, selected.feature_names)
    encoded_labels, class_names = encode_broad_labels(selected.labels, sample.class_order)

    def build_partition(positions: np.ndarray) -> PreparedPartition:
        return PreparedPartition(
            features=preprocessor.transform(selected.features[positions]),
            labels=encoded_labels[positions],
            row_indices=selected.row_indices[positions],
        )

    return PreparedDataset(
        train=build_partition(split.train),
        validation=build_partition(split.validation),
        test=build_partition(split.test),
        feature_names=selected.feature_names,
        class_names=class_names,
        preprocessor=preprocessor,
    )


def _negative_or_nonfinite_to_missing(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64).copy()
    values[(values < 0) | ~np.isfinite(values)] = np.nan
    return values


def _impute_in_place(values: np.ndarray, imputation_values: np.ndarray) -> None:
    missing_rows, missing_columns = np.where(np.isnan(values))
    values[missing_rows, missing_columns] = imputation_values[missing_columns]


def _validate_split_positions(sample_size: int, split: StratifiedSplit) -> None:
    all_positions = np.concatenate((split.train, split.validation, split.test))
    if len(all_positions) != sample_size or len(np.unique(all_positions)) != sample_size:
        raise PreprocessingError("Split positions must be complete and non-overlapping.")
    if all_positions.min() < 0 or all_positions.max() >= sample_size:
        raise PreprocessingError("Split positions are outside the sampled dataset.")
