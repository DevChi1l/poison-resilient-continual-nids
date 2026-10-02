"""Disk-backed, row-aligned preparation for the full combined flow collection.

Only labels, split codes, row IDs, and one float32 raw-feature matrix are
persisted. Feature batches, never the full feature matrix, enter RAM.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.lib.format import open_memmap

from .loader import inspect_parquet, iter_dataset_batches
from .preprocessing import FittedPreprocessor, PreprocessingError
from .sampling import BROAD_CLASS_ORDER


TRAIN, VALIDATION, TEST = 0, 1, 2
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class LargeDataPaths:
    root: Path
    labels: Path
    splits: Path
    row_ids: Path
    raw_features: Path


def large_data_paths(root: str | Path) -> LargeDataPaths:
    base = Path(root)
    return LargeDataPaths(base, base / "labels.npy", base / "splits.npy",
                          base / "row_ids.npy", base / "raw_features.npy")


def _new_array(path: Path, dtype: str, shape: tuple[int, ...]) -> np.memmap:
    return open_memmap(str(path) + ".partial", mode="w+", dtype=dtype, shape=shape)


def _finish_array(array: np.memmap, path: Path) -> None:
    array.flush()
    del array
    os.replace(str(path) + ".partial", path)


def _open_checked(path: Path, dtype: str, shape: tuple[int, ...]) -> np.memmap:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.dtype(dtype) or array.shape != shape:
        raise ValueError(f"Unexpected disk array format: {path}")
    return array


def verify_full_partitions(paths: LargeDataPaths, row_count: int) -> dict:
    """Check exact row coverage, split codes, global IDs, and per-class counts."""

    labels = _open_checked(paths.labels, "uint8", (row_count,))
    splits = _open_checked(paths.splits, "uint8", (row_count,))
    ids = _open_checked(paths.row_ids, "int64", (row_count,))
    counts = np.zeros((len(BROAD_CLASS_ORDER), 3), dtype=np.int64)
    for start in range(0, row_count, 262_144):
        end = min(start + 262_144, row_count)
        if not np.array_equal(ids[start:end], np.arange(start, end, dtype=np.int64)):
            raise ValueError("Original row IDs are incomplete, duplicated, or out of order")
        if np.any(labels[start:end] >= len(BROAD_CLASS_ORDER)) or np.any(splits[start:end] > TEST):
            raise ValueError("Invalid class ID or split code")
        counts += np.bincount(
            labels[start:end].astype(np.int64) * 3 + splits[start:end],
            minlength=len(BROAD_CLASS_ORDER) * 3,
        ).reshape(len(BROAD_CLASS_ORDER), 3)
    if int(counts.sum()) != row_count or np.any(counts == 0):
        raise ValueError("Incomplete or empty class-stratified partitions")
    index_paths = [paths.root / f"{name}_indices.npy" for name in SPLIT_NAMES]
    if any(path.exists() for path in index_paths) and not all(path.exists() for path in index_paths):
        raise ValueError("Incomplete persisted split-index files")
    for code, name in enumerate(SPLIT_NAMES):
        index_path = paths.root / f"{name}_indices.npy"
        if index_path.exists():
            saved = np.load(index_path, mmap_mode="r", allow_pickle=False)
            expected = np.flatnonzero(splits == code)
            if saved.dtype != np.int64 or not np.array_equal(saved, expected):
                raise ValueError(f"Persisted {name} original-row indices do not match split codes")
    return {
        "row_count": row_count,
        "class_order": list(BROAD_CLASS_ORDER),
        "counts": {name: {SPLIT_NAMES[j]: int(counts[i, j]) for j in range(3)}
                   for i, name in enumerate(BROAD_CLASS_ORDER)},
        "coverage": "each original row ID occurs exactly once, assigned one split",
    }


def prepare_full_partitions(
    parquet_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    batch_size: int = 65_536,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Persist deterministic 70/15/15 splits without reading feature columns."""

    info = inspect_parquet(parquet_path)
    paths = large_data_paths(output_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    split_manifest = paths.root / "split_manifest.json"
    if split_manifest.exists():
        saved = json.loads(split_manifest.read_text())
        if saved.get("seed") != seed or saved.get("row_count") != info.row_count:
            raise ValueError("Existing split manifest does not match input rows or seed")
        actual = verify_full_partitions(paths, info.row_count)
        if actual["counts"] != saved["counts"]:
            raise ValueError("Persisted split counts differ from manifest")
        return saved

    label_to_id = {name: i for i, name in enumerate(BROAD_CLASS_ORDER)}
    labels = _new_array(paths.labels, "uint8", (info.row_count,))
    offset = 0
    for batch in iter_dataset_batches(parquet_path, columns=("ClassLabel",), batch_size=batch_size):
        names = batch.column(0).to_pylist()
        try:
            labels[offset:offset + len(names)] = [label_to_id[name] for name in names]
        except KeyError as error:
            raise ValueError(f"Unexpected broad class: {error}") from error
        offset += len(names)
        if progress and (offset == info.row_count or offset % (batch_size * 32) == 0):
            progress(f"label scan {offset:,}/{info.row_count:,}")
    if offset != info.row_count:
        raise ValueError("Label scan ended before metadata row count")
    _finish_array(labels, paths.labels)
    labels = _open_checked(paths.labels, "uint8", (info.row_count,))

    splits = _new_array(paths.splits, "uint8", (info.row_count,))
    splits[:] = 255
    ids = _new_array(paths.row_ids, "int64", (info.row_count,))
    for start in range(0, info.row_count, 262_144):
        end = min(start + 262_144, info.row_count)
        ids[start:end] = np.arange(start, end, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for class_id, name in enumerate(BROAD_CLASS_ORDER):
        positions = np.flatnonzero(labels == class_id)
        if positions.size < 3:
            raise ValueError(f"Class {name} has fewer than three rows")
        exact = np.asarray((0.70, 0.15, 0.15)) * positions.size
        counts = np.floor(exact).astype(np.int64)
        for part in np.argsort(-(exact - counts), kind="stable")[:positions.size - int(counts.sum())]:
            counts[part] += 1
        shuffled = rng.permutation(positions)
        first, second = int(counts[0]), int(counts[0] + counts[1])
        splits[shuffled[:first]] = TRAIN
        splits[shuffled[first:second]] = VALIDATION
        splits[shuffled[second:]] = TEST
        if progress:
            progress(f"split {name}: {tuple(int(x) for x in counts)}")
    _finish_array(splits, paths.splits)
    _finish_array(ids, paths.row_ids)
    assigned = _open_checked(paths.splits, "uint8", (info.row_count,))
    for code, name in enumerate(SPLIT_NAMES):
        np.save(paths.root / f"{name}_indices.npy", np.flatnonzero(assigned == code))
    summary = verify_full_partitions(paths, info.row_count)
    summary.update({"seed": seed, "fractions": [0.70, 0.15, 0.15],
                    "split_codes": {"train": TRAIN, "validation": VALIDATION, "test": TEST}})
    split_manifest.write_text(json.dumps(summary, indent=2))
    return summary


def prepare_raw_feature_store(
    parquet_path: str | Path,
    output_dir: str | Path,
    *,
    batch_size: int = 65_536,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Write one row-aligned float32 feature matrix, never either target column."""

    info = inspect_parquet(parquet_path)
    paths = large_data_paths(output_dir)
    verify_full_partitions(paths, info.row_count)
    names = info.schema.retained_feature_names
    marker = paths.root / "features_manifest.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved.get("row_count") != info.row_count or saved.get("feature_names") != list(names):
            raise ValueError("Existing feature store has incompatible schema")
        _open_checked(paths.raw_features, "float32", (info.row_count, len(names)))
        return paths.raw_features

    features = _new_array(paths.raw_features, "float32", (info.row_count, len(names)))
    labels = _open_checked(paths.labels, "uint8", (info.row_count,))
    lookup = {name: i for i, name in enumerate(BROAD_CLASS_ORDER)}
    offset = 0
    for batch in iter_dataset_batches(
        parquet_path, batch_size=batch_size, columns=(*names, "ClassLabel")
    ):
        count = batch.num_rows
        for column in range(len(names)):
            raw = batch.column(column).to_numpy(zero_copy_only=False)
            with np.errstate(over="ignore", invalid="ignore"):
                values = np.asarray(raw, dtype=np.float32)
            features[offset:offset + count, column] = values
        chunk = features[offset:offset + count]
        chunk[(chunk < 0) | ~np.isfinite(chunk)] = np.nan
        actual_labels = np.fromiter(
            (lookup[name] for name in batch.column(len(names)).to_pylist()),
            dtype=np.uint8, count=count,
        )
        if not np.array_equal(actual_labels, labels[offset:offset + count]):
            raise ValueError("Feature batch/label row alignment failed")
        offset += count
        if progress and (offset == info.row_count or offset % (batch_size * 32) == 0):
            progress(f"feature store {offset:,}/{info.row_count:,}")
    if offset != info.row_count:
        raise ValueError("Feature scan ended before metadata row count")
    _finish_array(features, paths.raw_features)
    marker.write_text(json.dumps({"row_count": info.row_count, "feature_names": list(names),
                                  "dtype": "float32", "invalid_rule": "negative or non-finite -> NaN",
                                  "targets_excluded": ["Label", "ClassLabel"]}, indent=2))
    return paths.raw_features


def _fit_scope_mask(labels: np.ndarray, splits: np.ndarray, scope: str) -> np.ndarray:
    if scope not in ("static", "continual"):
        raise ValueError("scope must be static or continual")
    mask = splits == TRAIN
    if scope == "continual":
        mask &= labels <= 4
    return mask


def fit_disk_preprocessor(
    output_dir: str | Path,
    *,
    scope: str,
    reservoir_size: int = 200_000,
    seed: int = 42,
    batch_size: int = 65_536,
    progress: Callable[[str], None] | None = None,
) -> FittedPreprocessor:
    """Approximate medians from seeded training-only rows; stream exact moments."""

    if reservoir_size < 1 or batch_size < 1:
        raise ValueError("reservoir_size and batch_size must be positive")
    paths = large_data_paths(output_dir)
    meta = json.loads((paths.root / "features_manifest.json").read_text())
    n, width = meta["row_count"], len(meta["feature_names"])
    state_path = paths.root / f"preprocessing_{scope}.json"
    if state_path.exists():
        saved = json.loads(state_path.read_text())
        if saved["reservoir_seed"] != seed or saved["reservoir_limit"] != reservoir_size:
            raise ValueError("Existing preprocessing configuration differs")
        return FittedPreprocessor(tuple(saved["feature_names"]),
                                  np.asarray(saved["imputation_values"], dtype=np.float64),
                                  np.asarray(saved["means"], dtype=np.float64),
                                  np.asarray(saved["scales"], dtype=np.float64))
    features = _open_checked(paths.raw_features, "float32", (n, width))
    labels = _open_checked(paths.labels, "uint8", (n,))
    splits = _open_checked(paths.splits, "uint8", (n,))
    eligible = np.flatnonzero(_fit_scope_mask(labels, splits, scope))
    selected = np.sort(np.random.default_rng(seed).choice(
        eligible, size=min(reservoir_size, eligible.size), replace=False))
    if selected.size == 0:
        raise PreprocessingError("No training rows for preprocessing scope")
    reservoir = np.empty((selected.size, width), dtype=np.float32)
    for start in range(0, selected.size, batch_size):
        end = min(start + batch_size, selected.size)
        reservoir[start:end] = features[selected[start:end]]
    with np.errstate(all="ignore"):
        medians = np.nanmedian(reservoir.astype(np.float64), axis=0)
    if not np.isfinite(medians).all():
        raise PreprocessingError("At least one feature has no valid median in the training reservoir")
    del reservoir, eligible, selected
    total = 0
    mean = np.zeros(width, dtype=np.float64)
    m2 = np.zeros(width, dtype=np.float64)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        selected_rows = _fit_scope_mask(labels[start:end], splits[start:end], scope)
        if not selected_rows.any():
            continue
        values = np.asarray(features[start:end][selected_rows], dtype=np.float64)
        missing = np.isnan(values)
        values[missing] = medians[np.where(missing)[1]]
        chunk_n = len(values)
        chunk_mean = values.mean(axis=0)
        chunk_m2 = np.square(values - chunk_mean).sum(axis=0)
        delta = chunk_mean - mean
        combined = total + chunk_n
        mean += delta * chunk_n / combined
        m2 += chunk_m2 + delta * delta * total * chunk_n / combined
        total = combined
        if progress and (end == n or end % (batch_size * 32) == 0):
            progress(f"{scope} moment fit scanned {end:,}/{n:,}; used {total:,}")
    scales = np.sqrt(m2 / total)
    scales[scales == 0] = 1.0
    state = FittedPreprocessor(tuple(meta["feature_names"]), medians, mean, scales)
    record = {"scope": scope, "fit_rows": total, "reservoir_limit": reservoir_size,
              "reservoir_size_used": min(reservoir_size, total), "reservoir_seed": seed,
              "median_method": "seeded uniform without-replacement training-only reservoir; approximate",
              "moments_method": "streaming population mean/std after median imputation",
              "feature_names": list(state.feature_names),
              "imputation_values": medians.tolist(), "means": mean.tolist(),
              "scales": scales.tolist()}
    state_path.write_text(json.dumps(record, indent=2))
    return state
