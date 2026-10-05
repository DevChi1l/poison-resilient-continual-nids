"""Bounded upload parsing, strict schema validation, and CPU inference."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.preprocessing import FittedPreprocessor
from src.evaluation.classification import classification_metrics

from .artifacts import (
    ArtifactConfigurationError,
    ModelArtifact,
    load_fitted_preprocessor,
    verify_file,
)


class UploadSchemaError(ValueError):
    """Raised when an uploaded table is unsafe or incompatible with a model."""


@dataclass(frozen=True)
class PreparedUpload:
    transformed_features: np.ndarray
    labels: np.ndarray | None
    original_frame: pd.DataFrame
    label_column: str | None


@dataclass(frozen=True)
class InferenceResult:
    predictions: np.ndarray
    probabilities: np.ndarray
    metrics: dict[str, Any] | None


def read_uploaded_table(content: bytes, filename: str, *, max_rows: int) -> pd.DataFrame:
    """Read at most the configured row bound from CSV or Parquet bytes."""

    if not content:
        raise UploadSchemaError("Uploaded file is empty")
    suffix = Path(filename).suffix.lower()
    try:
        if suffix == ".csv":
            frame = pd.read_csv(BytesIO(content), nrows=max_rows + 1)
        elif suffix in (".parquet", ".pq"):
            try:
                import pyarrow.parquet as parquet
            except ImportError as error:  # pragma: no cover - dependency message
                raise UploadSchemaError("PyArrow is required for Parquet uploads") from error
            parquet_file = parquet.ParquetFile(BytesIO(content))
            if parquet_file.metadata.num_rows > max_rows:
                raise UploadSchemaError(
                    f"Upload has {parquet_file.metadata.num_rows:,} rows; limit is {max_rows:,}"
                )
            batches = list(parquet_file.iter_batches(batch_size=max_rows + 1))
            if not batches:
                frame = pd.DataFrame()
            elif len(batches) == 1:
                frame = batches[0].to_pandas()
            else:
                frame = pd.concat([batch.to_pandas() for batch in batches], ignore_index=True)
        else:
            raise UploadSchemaError("Upload must be CSV or Parquet")
    except UploadSchemaError:
        raise
    except Exception as error:
        raise UploadSchemaError(f"Could not read {suffix or 'upload'}: {error}") from error
    if len(frame) == 0:
        raise UploadSchemaError("Upload contains no rows")
    if len(frame) > max_rows:
        raise UploadSchemaError(f"Upload exceeds the {max_rows:,}-row CPU inference limit")
    return frame


def _encode_labels(series: pd.Series, model: ModelArtifact) -> np.ndarray:
    name_to_id = {name: class_id for class_id, name in model.class_names.items()}
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.isna().any() or not np.equal(numeric, np.floor(numeric)).all():
            raise UploadSchemaError("Label column must contain integer IDs or exact class names")
        labels = numeric.to_numpy(dtype=np.int64)
    else:
        unknown = sorted(set(series.astype(str)) - set(name_to_id))
        if unknown:
            raise UploadSchemaError("Unknown labelled classes: " + ", ".join(unknown))
        labels = np.asarray([name_to_id[str(value)] for value in series], dtype=np.int64)
    if not np.isin(labels, model.class_ids).all():
        raise UploadSchemaError("Label IDs are outside the selected model's class mapping")
    return labels


def prepare_upload(
    frame: pd.DataFrame,
    model: ModelArtifact,
    preprocessor: FittedPreprocessor,
    *,
    max_rows: int,
    label_column: str | None = None,
) -> PreparedUpload:
    """Require exact raw feature order, then apply the model's paired state."""

    if len(frame) == 0 or len(frame) > max_rows:
        raise UploadSchemaError(f"Input must contain 1 to {max_rows:,} rows")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise UploadSchemaError("Duplicate columns: " + ", ".join(map(str, duplicates)))
    if label_column is not None and label_column not in frame.columns:
        raise UploadSchemaError(f"Label column {label_column!r} is absent")
    feature_columns = tuple(column for column in frame.columns if column != label_column)
    expected = preprocessor.feature_names
    if feature_columns != expected:
        missing = [name for name in expected if name not in feature_columns]
        unexpected = [str(name) for name in feature_columns if name not in expected]
        if not missing and not unexpected:
            detail = "feature names match but their order differs"
        else:
            detail = f"missing={missing}; unexpected={unexpected}"
        raise UploadSchemaError(
            f"Feature schema must exactly match the configured {len(expected)}-feature order: {detail}"
        )
    try:
        raw = frame.loc[:, expected].apply(pd.to_numeric, errors="raise").to_numpy(
            dtype=np.float64
        )
    except (TypeError, ValueError) as error:
        raise UploadSchemaError(f"All configured features must be numeric: {error}") from error
    transformed = preprocessor.transform(raw)
    if not np.isfinite(transformed).all():
        raise UploadSchemaError("Preprocessing produced non-finite values")
    labels = _encode_labels(frame[label_column], model) if label_column else None
    return PreparedUpload(transformed, labels, frame.copy(), label_column)


def load_verified_model(model: ModelArtifact):
    """Load one real project checkpoint on CPU and verify its public contract."""

    verify_file(model.checkpoint, model.checkpoint_sha256, label=model.display_name)
    try:
        from src.models import TabularTransformerClassifier
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ArtifactConfigurationError(
            "PyTorch is required for real checkpoint inference; install requirements-demo.txt"
        ) from error
    classifier = TabularTransformerClassifier.load(model.checkpoint, device="cpu")
    preprocessor = load_fitted_preprocessor(model)
    if classifier.config.num_features != len(preprocessor.feature_names):
        raise ArtifactConfigurationError("Checkpoint width does not match preprocessing state")
    if tuple(classifier.class_ids) != model.class_ids:
        raise ArtifactConfigurationError("Checkpoint class IDs do not match configured mapping")
    return classifier, preprocessor


def run_inference(
    classifier: Any,
    prepared: PreparedUpload,
    model: ModelArtifact,
) -> InferenceResult:
    """Run bounded inference and optional known-class metrics; never train."""

    probabilities = np.asarray(classifier.predict_proba(prepared.transformed_features))
    if probabilities.shape != (len(prepared.transformed_features), len(model.class_ids)):
        raise RuntimeError("Model returned a probability array with an unexpected shape")
    columns = np.asarray(model.class_ids, dtype=np.int64)
    predictions = columns[probabilities.argmax(axis=1)]
    metrics = None
    if prepared.labels is not None:
        metrics = classification_metrics(
            prepared.labels,
            predictions,
            class_ids=model.class_ids,
            class_names=model.class_names,
            benign_class_id=0,
        )
    return InferenceResult(predictions, probabilities, metrics)
