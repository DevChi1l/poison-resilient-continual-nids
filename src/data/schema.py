"""Schema contract for the local combined flow collection.

This module deliberately validates names and types only. Cleaning, sampling,
splitting, and fitting preprocessing state are separate later steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import pyarrow as pa
except ImportError:  # pragma: no cover - exercised only without an optional dependency
    pa = None  # type: ignore[assignment]


TARGET_COLUMNS = ("Label", "ClassLabel")
"""Fine-grained and broad target columns stored in the raw Parquet file."""

DROPPED_FEATURES = (
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Seg Size Min",
)
"""Features excluded by the audited Review-2 baseline policy."""

EXPECTED_NUMERIC_FEATURE_COUNT = 57


class DatasetSchemaError(ValueError):
    """Raised when a Parquet table does not meet the documented data contract."""


@dataclass(frozen=True)
class DatasetSchema:
    """Validated raw and retained feature names in their original file order."""

    all_columns: tuple[str, ...]
    raw_numeric_features: tuple[str, ...]
    retained_feature_names: tuple[str, ...]
    target_columns: tuple[str, ...]


def _require_pyarrow() -> Any:
    if pa is None:
        raise ImportError(
            "PyArrow is required for Parquet loading. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        )
    return pa


def validate_schema(arrow_schema: Any, *, strict: bool = True) -> DatasetSchema:
    """Validate the combined-flow schema and return deterministic feature order.

    Strict validation is for the project dataset: it requires 57 numeric features
    and all documented dropped-feature names. Tests can disable it for small
    synthetic Parquet files while still checking target types and feature selection.
    """

    arrow = _require_pyarrow()
    fields = list(arrow_schema)
    all_columns = tuple(field.name for field in fields)
    field_by_name = {field.name: field for field in fields}

    missing_targets = [name for name in TARGET_COLUMNS if name not in field_by_name]
    if missing_targets:
        raise DatasetSchemaError(
            "Missing required target columns: " + ", ".join(missing_targets)
        )

    for name in TARGET_COLUMNS:
        data_type = field_by_name[name].type
        if not (arrow.types.is_string(data_type) or arrow.types.is_large_string(data_type)):
            raise DatasetSchemaError(
                f"Target column {name!r} must be a string, found {data_type}."
            )

    raw_numeric_features = tuple(
        field.name
        for field in fields
        if arrow.types.is_integer(field.type) or arrow.types.is_floating(field.type)
    )
    if not raw_numeric_features:
        raise DatasetSchemaError("No numeric flow features were found.")

    missing_dropped = [name for name in DROPPED_FEATURES if name not in raw_numeric_features]
    if strict and missing_dropped:
        raise DatasetSchemaError(
            "Missing audited dropped-feature columns: " + ", ".join(missing_dropped)
        )
    if strict and len(raw_numeric_features) != EXPECTED_NUMERIC_FEATURE_COUNT:
        raise DatasetSchemaError(
            f"Expected {EXPECTED_NUMERIC_FEATURE_COUNT} numeric features, found "
            f"{len(raw_numeric_features)}."
        )

    retained_feature_names = tuple(
        name for name in raw_numeric_features if name not in DROPPED_FEATURES
    )
    if not retained_feature_names:
        raise DatasetSchemaError("No retained numeric features remain after exclusions.")

    return DatasetSchema(
        all_columns=all_columns,
        raw_numeric_features=raw_numeric_features,
        retained_feature_names=retained_feature_names,
        target_columns=TARGET_COLUMNS,
    )
