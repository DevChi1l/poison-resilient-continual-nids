# Developer B Preprocessing History and Handoff

> Historical origin record. The data package is now owned by the unified A+B+C
> team; lettered handoff labels below describe the sequence in which the code
> was created, not current ownership.

## Purpose

This document records the complete Review-2 preprocessing work for the local combined flow collection. It distinguishes verified observations from proposed or unrun work.

The purpose of preprocessing is to turn raw network-flow records into reproducible numeric matrices that a tabular Transformer can consume safely, without changing the raw dataset or leaking validation/test information into training.

## Raw input dataset

- Local path: `data/cic-collection.parquet`.
- Raw file status: ignored by Git and unchanged by every operation described here.
- File size: 1,026,513,455 bytes (about 979 MiB).
- SHA-256: `666af53c788c79312b421c607289cb555c2335ae51ba39b2700a357595fa8dba`.
- Format: Apache Parquet, written by PyArrow 8.0.0.
- Rows: 9,167,581.
- Columns: 59.
- Row groups: 1.

The file contains no source-dataset identifier, timestamp, IP address, port, flow ID, or raw packet payload. Its individual rows therefore cannot be attributed to CIC-IDS2017, CIC-IDS2018, or another source based on the file contents alone. The project calls it the **combined flow collection**.

## Raw schema and labels

The table has 57 numeric flow features and two textual target columns.

| Column | Role | Observed state |
| --- | --- | --- |
| `Label` | Fine-grained target | 33 labels, no missing values |
| `ClassLabel` | Broad target | 8 labels, no missing values |
| Remaining 57 columns | Numeric flow features | Network-flow duration, packet, rate, timing, flag, size, active, and idle summaries |

For the first Review-2 baseline, the selected training target is `ClassLabel`. It gives eight broad classes that are large enough for a class-balanced two-task demonstration. `Label` is preserved as metadata for later fine-grained experiments.

The stable broad-class mapping is:

| Integer ID | `ClassLabel` |
| ---: | --- |
| 0 | Benign |
| 1 | DDoS |
| 2 | DoS |
| 3 | Botnet |
| 4 | Bruteforce |
| 5 | Infiltration |
| 6 | Webattack |
| 7 | Portscan |

This mapping is fixed by `src/data/sampling.py` and must be kept aligned with a model's output logits and `predict_proba` columns.

## Audit method

The raw 979 MiB file was never loaded in full into memory. It was read using PyArrow in batches:

```python
for batch in parquet_file.iter_batches(
    batch_size=65_536,
    columns=selected_columns,
):
    # inspect or process one bounded batch
```

This approach was used to:

1. Read Parquet metadata for row count, schema, and column types.
2. Count `Label` and `ClassLabel` values across all rows.
3. Inspect nulls, NaNs, infinities, constant columns, minima, maxima, and negative values in all numeric features.
4. Count overlap between invalid-value patterns and broad classes.

## Data-quality findings

No numeric feature contains null values, NaN, positive infinity, negative infinity, or only one constant value.

The audit found negative values in fields that represent durations, rates, time intervals, header sizes, segment sizes, or initial window bytes. These values are stored in the original file; they are not a Parquet-reading error. Direct row samples confirmed values such as:

```text
Flow Duration = -1
Flow Bytes/s = -12,000,000.0
Fwd Header Length = -1,929,349,973
Fwd Seg Size Min = -83,885,313
Init Fwd Win Bytes = -1
```

| Feature | Negative values | Percent of raw rows |
| --- | ---: | ---: |
| Flow Duration | 96 | 0.0010% |
| Flow Bytes/s | 53 | 0.0006% |
| Flow Packets/s | 96 | 0.0010% |
| Flow IAT Mean | 96 | 0.0010% |
| Flow IAT Max | 85 | 0.0009% |
| Flow IAT Min | 2,816 | 0.0307% |
| Fwd IAT Total | 14 | 0.0002% |
| Fwd IAT Mean | 14 | 0.0002% |
| Fwd IAT Max | 3 | 0.0000% |
| Fwd IAT Min | 32 | 0.0003% |
| Fwd Header Length | 50,907 | 0.5553% |
| Bwd Header Length | 257 | 0.0028% |
| Init Fwd Win Bytes | 2,658,795 | 29.0021% |
| Init Bwd Win Bytes | 3,766,505 | 41.0850% |
| Fwd Seg Size Min | 74,142 | 0.8087% |

The two initial-window fields use only `-1` for their negative values, indicating a likely unavailable-value sentinel. The most problematic header-length and minimum-segment-size values are highly concentrated in DDoS records. There are 77,114 raw rows with at least one non-window negative anomaly, and 73,860 of them are DDoS. Deleting all affected rows would disproportionately remove DDoS evidence, so row deletion was not selected.

## What was removed and what was retained

### Raw data changes

Nothing was removed or modified in the original Parquet file.

- Raw rows removed: `0`.
- Raw columns deleted: `0`.
- Raw values edited: `0`.
- Processed-data files written: `0`.

### Model-input feature selection

Three features are excluded only from the Review-2 model input because they contain systematic, class-concentrated invalid values:

```text
Fwd Header Length
Bwd Header Length
Fwd Seg Size Min
```

This leaves 54 retained numeric features. The original three columns remain in the raw file.

## Sampling and splitting

The code uses a deterministic, class-balanced reservoir sampler.

```text
Read ClassLabel in batches
    -> retain only selected original row indices
    -> split sampled positions by class
    -> materialize only requested raw rows later
```

Reservoir sampling means every row in a requested class has equal probability of being selected, while memory remains bounded. The raw file is not copied during selection.

The default split is stratified by `ClassLabel`:

```text
70% train
15% validation
15% test
```

Each class is independently shuffled using a seed, then allocated to the three partitions. This ensures every selected row belongs to exactly one partition and each class appears in each partition.

Observed smoke selection:

| Setting | Value |
| --- | --- |
| Classes | All eight broad classes |
| Rows per class | 64 |
| Sampling seed | 42 |
| Split seed | 42 |
| Total selected rows | 512 |
| Train | 360 rows, 45 per class |
| Validation | 80 rows, 10 per class |
| Test | 72 rows, 9 per class |

The smoke sample proves the pipeline. It is not the intended T4 training scale. The planned development/Colab configuration is 2,000 rows per class: 16,000 total selected rows, with 1,400/300/300 train/validation/test rows per class. A larger class-capped Colab configuration can also be selected through the same interface.

## Preprocessing algorithm

After sampling and splitting, preprocessing operates only in memory:

```text
Selected raw feature matrix
    -> convert every negative or non-finite retained value to missing
    -> calculate a per-feature median from training rows only
    -> replace missing values in all partitions with those training medians
    -> calculate mean and standard deviation from imputed training rows only
    -> standardize train, validation, and test with those training values
    -> convert outputs to float32
```

Formally, for retained feature value `x`:

```text
x < 0 or x is non-finite  -> missing
missing                     -> training median for that feature
standardized x              -> (x - training mean) / training standard deviation
```

If a feature has zero standard deviation in the selected training sample, its scale is set to `1.0`; this prevents division by zero while preserving the centered value.

The preprocessor is fit exclusively on training features. Validation and test data are transformed with the already-fitted training medians, means, and standard deviations. This prevents information leakage from validation/test rows into model training.

## Observed preprocessing result

The 512-row smoke sample was passed through the actual preprocessing code.

| Partition | Matrix shape | Dtype | Values finite? | Broad labels per class |
| --- | --- | --- | --- | ---: |
| Train | `(360, 54)` | `float32` | Yes | 45 |
| Validation | `(80, 54)` | `float32` | Yes | 10 |
| Test | `(72, 54)` | `float32` | Yes | 9 |

After standardization, the training matrix had:

```text
largest absolute per-feature mean: 5.5e-7
minimum per-feature standard deviation: 0.9999972
maximum per-feature standard deviation: 1.0000035
```

These are effectively mean `0` and standard deviation `1`; small deviations result from converting arrays to `float32`.

Negative values may appear in the final standardized matrices. This is correct: after centering, a negative standardized value means the cleaned feature is below its training mean. It does not mean the original invalid negative value remains.

## Implementation files

| File | Responsibility |
| --- | --- |
| `src/data/schema.py` | Raw schema contract, target names, three excluded features |
| `src/data/loader.py` | Metadata inspection and bounded Parquet batch reading |
| `src/data/sampling.py` | Stable broad-class order and class-balanced reservoir sampling |
| `src/data/splitting.py` | Seeded, stratified 70/15/15 split positions |
| `src/data/preprocessing.py` | Selected-row materialization, training-only imputation/scaling, label encoding |
| `tests/test_data_loader.py` | Seven synthetic smoke tests |
| `data/metadata/combined_flow_collection_audit.md` | Dataset provenance limitation, audit observations, and policy |

## Model integration example

The team can use the data package now. There are no saved prepared-data files
yet, so the training entry point should import the functions and construct the
selected dataset.

```python
from src.data import (
    prepare_sampled_dataset,
    sample_class_balanced_indices,
    stratified_split_indices,
)

sample = sample_class_balanced_indices(
    "data/cic-collection.parquet",
    per_class_limit=2_000,
    seed=42,
)
split = stratified_split_indices(sample.labels, seed=42)
dataset = prepare_sampled_dataset(
    "data/cic-collection.parquet",
    sample,
    split,
)

X_train, y_train = dataset.train.features, dataset.train.labels
X_val, y_val = dataset.validation.features, dataset.validation.labels
X_test, y_test = dataset.test.features, dataset.test.labels
```

The resulting contract is:

```python
X_train: float32 array, shape (n_train, 54)
y_train: int64 array, shape (n_train,)
X_val:   float32 array, shape (n_validation, 54)
y_val:   int64 array, shape (n_validation,)
X_test:  float32 array, shape (n_test, 54)
y_test:  int64 array, shape (n_test,)

dataset.feature_names: tuple[str, ...], length 54
dataset.class_names: dict[int, str], IDs 0 through 7
dataset.preprocessor: fitted training-only medians, means, and scales
```

The model integration must keep `dataset.class_names` aligned with the model
head and probabilities. In particular, probability column `i` must correspond
to `dataset.class_names[i]`.

For a future inference/demo run, reuse the same `dataset.preprocessor` state. Do not fit another preprocessor on validation, test, or inference data.

## GitHub and artifact policy

Push the Python code, tests, dependency manifest, Markdown documentation, and small configuration files after review. Do **not** push:

- `data/cic-collection.parquet` or any raw dataset.
- A full processed 16,000-row or 300,000-row dataset unless the team explicitly chooses a storage policy and verifies file size/licensing.
- Model checkpoints, bulk results, or machine-specific local configuration.

The raw file is already ignored by `.gitignore`. If the team later needs materialized prepared datasets to avoid repeated preprocessing in Colab, save them under an ignored location such as `data/processed/`, include the raw-file checksum, selection seed, split seed, feature list, and preprocessor metadata, and share them through a suitable external storage service rather than ordinary Git.

## Verified and not yet verified

Verified:

- Full raw-file schema and label audit.
- Invalid-value audit and class-overlap analysis.
- Strict batch loader on the real file.
- Class-balanced sampling and stratified splits.
- Train-only preprocessing on the real 512-row smoke configuration.
- Seven synthetic data tests.

Not yet verified:

- 2,000-per-class development preprocessing run.
- Large Colab/T4 training run.
- Real-data model training (the source-level model/data integration exists).
- Task 1/Task 2 class-incremental training.
- Held-out-class novelty evaluation, poisoning, replay, and evaluation experiments.
