"""Prediction-independent export of a bounded held-out flow sample."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.data.loader import inspect_parquet, iter_dataset_batches
from src.data.preprocessing import materialize_selected_rows

from .artifacts import (
    ArtifactConfigurationError,
    DemoArtifactConfig,
    file_sha256,
    load_fitted_preprocessor,
    verify_file,
)


class SampleExportError(ValueError):
    """Raised when a held-out rehearsal sample cannot be exported safely."""


@dataclass(frozen=True)
class SampleExportResult:
    flow_csv: Path
    provenance_csv: Path
    manifest_json: Path
    row_count: int
    class_counts: dict[str, int]
    source_row_ids: np.ndarray


def load_saved_test_indices(path: Path, *, row_count: int) -> np.ndarray:
    """Memory-map and validate persisted original-row test indices."""

    try:
        indices = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise SampleExportError(f"Could not read saved test indices {path}: {error}") from error
    if indices.dtype != np.dtype("int64") or indices.ndim != 1 or indices.size == 0:
        raise SampleExportError("Saved test indices must be a non-empty int64 vector")
    if indices[0] < 0 or indices[-1] >= row_count:
        raise SampleExportError("Saved test indices are outside the source Parquet row range")
    if np.any(indices[1:] <= indices[:-1]):
        raise SampleExportError("Saved test indices must be unique and strictly increasing")
    return indices


def select_class_stratified_rows(
    parquet_path: Path,
    test_indices: np.ndarray,
    *,
    class_names: tuple[str, ...],
    rows_per_class: int,
    seed: int,
    batch_size: int = 65_536,
    progress: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Reservoir-sample each class from held-out rows without model predictions."""

    if rows_per_class < 1:
        raise SampleExportError("rows_per_class must be positive")
    class_set = set(class_names)
    reservoirs: dict[str, list[int]] = {name: [] for name in class_names}
    seen = Counter({name: 0 for name in class_names})
    random = {
        name: np.random.default_rng(np.random.SeedSequence([seed, class_id]))
        for class_id, name in enumerate(class_names)
    }
    offset = 0
    matched = 0
    for batch in iter_dataset_batches(
        parquet_path,
        columns=("ClassLabel",),
        batch_size=batch_size,
        strict_schema=True,
    ):
        end = offset + batch.num_rows
        left = int(np.searchsorted(test_indices, offset, side="left"))
        right = int(np.searchsorted(test_indices, end, side="left"))
        if left != right:
            row_ids = np.asarray(test_indices[left:right], dtype=np.int64)
            local = row_ids - offset
            labels = batch.column(0).to_pylist()
            for row_id, local_index in zip(row_ids, local):
                label = str(labels[int(local_index)])
                if label not in class_set:
                    raise SampleExportError(f"Unexpected broad class in test split: {label}")
                seen[label] += 1
                reservoir = reservoirs[label]
                if len(reservoir) < rows_per_class:
                    reservoir.append(int(row_id))
                else:
                    replacement = int(random[label].integers(0, seen[label]))
                    if replacement < rows_per_class:
                        reservoir[replacement] = int(row_id)
            matched += right - left
        offset = end
        if progress and (offset % (batch_size * 32) == 0):
            progress(f"label scan {offset:,} rows; matched {matched:,} saved test rows")
        if matched == len(test_indices):
            break
    if matched != len(test_indices):
        raise SampleExportError(
            f"Matched {matched:,} saved test rows, expected {len(test_indices):,}"
        )
    absent = [name for name in class_names if seen[name] == 0]
    if absent:
        raise SampleExportError("Test split has no examples for: " + ", ".join(absent))

    selected: list[int] = []
    counts: dict[str, int] = {}
    for name in class_names:
        class_rows = sorted(reservoirs[name])
        selected.extend(class_rows)
        counts[name] = len(class_rows)
    return np.asarray(selected, dtype=np.int64), counts


def _write_csv_atomic(frame: pd.DataFrame, destination: Path) -> None:
    partial = destination.with_name(destination.name + ".partial")
    frame.to_csv(partial, index=False)
    os.replace(partial, destination)


def export_rehearsal_sample(
    config: DemoArtifactConfig,
    output_dir: str | Path,
    *,
    rows_per_class: int = 5,
    seed: int = 42,
    batch_size: int = 65_536,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> SampleExportResult:
    """Export raw held-out flows plus separate source-row provenance."""

    if config.dataset is None:
        raise SampleExportError("Demo config has no dataset provenance section")
    dataset = config.dataset
    static_model = config.models[config.default_model]
    if static_model.preprocessing_scope != "static" or len(static_model.class_ids) != 8:
        raise SampleExportError("default_model must be the eight-class static model")
    if rows_per_class * len(static_model.class_ids) > config.max_inference_rows:
        raise SampleExportError("Requested sample exceeds the configured inference row limit")

    verify_file(dataset.source_parquet, dataset.source_sha256, label="source Parquet")
    verify_file(dataset.split_manifest, dataset.split_manifest_sha256, label="split manifest")
    verify_file(dataset.split_codes, dataset.split_codes_sha256, label="split-code array")
    verify_file(dataset.test_indices, dataset.test_indices_sha256, label="test-index array")
    info = inspect_parquet(dataset.source_parquet, strict_schema=True)
    state = load_fitted_preprocessor(static_model)
    if tuple(info.schema.retained_feature_names) != state.feature_names:
        raise SampleExportError(
            "Source Parquet feature order does not match the static preprocessor"
        )
    try:
        split_manifest = json.loads(dataset.split_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SampleExportError(f"Could not read split manifest: {error}") from error
    class_names = tuple(static_model.class_names.values())
    if split_manifest.get("row_count") != info.row_count:
        raise SampleExportError("Split manifest row count does not match source Parquet")
    if tuple(split_manifest.get("class_order", ())) != class_names:
        raise SampleExportError("Split manifest class order does not match static model")

    test_indices = load_saved_test_indices(dataset.test_indices, row_count=info.row_count)
    expected_test_rows = sum(
        int(split_manifest["counts"][name]["test"]) for name in class_names
    )
    if len(test_indices) != expected_test_rows:
        raise SampleExportError(
            "Saved test-index count does not match the recorded split manifest"
        )
    selected_ids, class_counts = select_class_stratified_rows(
        dataset.source_parquet,
        test_indices,
        class_names=class_names,
        rows_per_class=rows_per_class,
        seed=seed,
        batch_size=batch_size,
        progress=progress,
    )
    selected = materialize_selected_rows(
        dataset.source_parquet,
        selected_ids,
        batch_size=batch_size,
        strict_schema=True,
    )
    if selected.feature_names != state.feature_names:
        raise SampleExportError("Materialized feature order differs from static preprocessor")

    output = Path(output_dir).expanduser().resolve()
    flow_path = output / "heldout_flows.csv"
    provenance_path = output / "heldout_flow_provenance.csv"
    manifest_path = output / "heldout_flow_manifest.json"
    for path in (flow_path, provenance_path, manifest_path):
        if path.exists() and not overwrite:
            raise SampleExportError(f"Refusing to overwrite existing output: {path}")
    output.mkdir(parents=True, exist_ok=True)

    flow = pd.DataFrame(selected.features, columns=state.feature_names)
    flow["ClassLabel"] = selected.labels.astype(str)
    provenance = pd.DataFrame(
        {
            "sample_row": np.arange(len(selected_ids), dtype=np.int64),
            "source_row_id": selected_ids,
            "ClassLabel": selected.labels.astype(str),
        }
    )
    _write_csv_atomic(flow, flow_path)
    _write_csv_atomic(provenance, provenance_path)
    manifest = {
        "purpose": "bounded local flow-inference rehearsal; not a full-test estimate",
        "selection": {
            "split": "saved full-data test partition",
            "method": "seeded per-class reservoir sampling from saved test row indices",
            "seed": seed,
            "rows_per_class_limit": rows_per_class,
            "prediction_independent": True,
            "class_counts": class_counts,
        },
        "source": {
            "parquet": str(dataset.source_parquet),
            "parquet_sha256": dataset.source_sha256,
            "test_indices": str(dataset.test_indices),
            "test_indices_sha256": dataset.test_indices_sha256,
            "split_manifest_sha256": dataset.split_manifest_sha256,
            "split_codes_sha256": dataset.split_codes_sha256,
        },
        "schema": {
            "feature_count": len(state.feature_names),
            "feature_names": list(state.feature_names),
            "label_column": "ClassLabel",
            "source_row_ids_stored_separately": True,
        },
        "outputs": {
            "flow_csv": flow_path.name,
            "flow_csv_sha256": file_sha256(flow_path),
            "provenance_csv": provenance_path.name,
            "provenance_csv_sha256": file_sha256(provenance_path),
        },
        "limitations": [
            "Rows were not selected using predictions or correctness.",
            "Any metrics on this bounded rehearsal sample are not full-test performance.",
        ],
    }
    partial = manifest_path.with_name(manifest_path.name + ".partial")
    partial.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, manifest_path)
    return SampleExportResult(
        flow_csv=flow_path,
        provenance_csv=provenance_path,
        manifest_json=manifest_path,
        row_count=len(flow),
        class_counts=class_counts,
        source_row_ids=selected_ids.copy(),
    )
