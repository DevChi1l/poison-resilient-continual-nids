"""Data loading and schema-validation interfaces owned by Developer B."""

from .loader import DatasetInfo, inspect_parquet, iter_dataset_batches
from .sampling import (
    BROAD_CLASS_ORDER,
    ClassBalancedSample,
    SamplingError,
    sample_class_balanced_indices,
)
from .preprocessing import (
    FittedPreprocessor,
    PreparedDataset,
    PreparedPartition,
    PreprocessingError,
    SelectedRows,
    encode_broad_labels,
    fit_preprocessor,
    materialize_selected_rows,
    prepare_sampled_dataset,
)
from .schema import (
    DROPPED_FEATURES,
    TARGET_COLUMNS,
    DatasetSchema,
    DatasetSchemaError,
    validate_schema,
)
from .splitting import SplitError, StratifiedSplit, stratified_split_indices

__all__ = [
    "DROPPED_FEATURES",
    "BROAD_CLASS_ORDER",
    "TARGET_COLUMNS",
    "ClassBalancedSample",
    "FittedPreprocessor",
    "PreparedDataset",
    "PreparedPartition",
    "DatasetInfo",
    "DatasetSchema",
    "DatasetSchemaError",
    "PreprocessingError",
    "SamplingError",
    "SelectedRows",
    "SplitError",
    "StratifiedSplit",
    "inspect_parquet",
    "iter_dataset_batches",
    "encode_broad_labels",
    "fit_preprocessor",
    "materialize_selected_rows",
    "prepare_sampled_dataset",
    "sample_class_balanced_indices",
    "stratified_split_indices",
    "validate_schema",
]
