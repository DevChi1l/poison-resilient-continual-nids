"""Memory-safe inspection and batch loading for the combined flow collection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - exercised only without an optional dependency
    pq = None  # type: ignore[assignment]

from .schema import DatasetSchema, DatasetSchemaError, validate_schema


@dataclass(frozen=True)
class DatasetInfo:
    """Immutable metadata needed to verify the raw file before processing it."""

    path: Path
    row_count: int
    column_count: int
    row_group_count: int
    schema: DatasetSchema


def _require_pyarrow() -> Any:
    if pq is None:
        raise ImportError(
            "PyArrow is required for Parquet loading. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        )
    return pq


def inspect_parquet(path: str | Path, *, strict_schema: bool = True) -> DatasetInfo:
    """Read Parquet metadata and validate the schema without loading table rows."""

    parquet = _require_pyarrow()
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Parquet dataset was not found: {dataset_path}")

    parquet_file = parquet.ParquetFile(dataset_path)
    metadata = parquet_file.metadata
    return DatasetInfo(
        path=dataset_path,
        row_count=metadata.num_rows,
        column_count=metadata.num_columns,
        row_group_count=metadata.num_row_groups,
        schema=validate_schema(parquet_file.schema_arrow, strict=strict_schema),
    )


def iter_dataset_batches(
    path: str | Path,
    *,
    batch_size: int = 65_536,
    strict_schema: bool = True,
    columns: Sequence[str] | None = None,
) -> Iterator[Any]:
    """Yield selected Parquet rows in bounded batches.

    The default yields the 54 retained Review-2 features and both target columns.
    Callers can request a validated subset for targeted audits. The function never
    loads the full 9-million-row dataset into memory.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    parquet = _require_pyarrow()
    info = inspect_parquet(path, strict_schema=strict_schema)
    allowed_columns = set(info.schema.all_columns)
    selected_columns = tuple(columns) if columns is not None else (
        info.schema.retained_feature_names + info.schema.target_columns
    )

    unknown_columns = [name for name in selected_columns if name not in allowed_columns]
    if unknown_columns:
        raise DatasetSchemaError(
            "Requested columns are absent from the dataset: " + ", ".join(unknown_columns)
        )

    parquet_file = parquet.ParquetFile(info.path)
    yield from parquet_file.iter_batches(batch_size=batch_size, columns=selected_columns)
