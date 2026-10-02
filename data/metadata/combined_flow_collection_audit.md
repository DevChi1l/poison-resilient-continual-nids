# Combined Flow Collection: Audit and Review-2 Data Decision

## Scope and provenance

This card documents read-only observations for the local file `data/cic-collection.parquet`. The file is ignored by Git and must not be committed or pushed.

- File size: 1,026,513,455 bytes (about 979 MiB).
- SHA-256: `666af53c788c79312b421c607289cb555c2335ae51ba39b2700a357595fa8dba`.
- Format: Apache Parquet, written by PyArrow 8.0.0.
- Rows: 9,167,581.
- Columns: 59, stored in one row group.
- Per-row source-dataset identifiers are absent. The file is therefore referred to as a combined flow collection. It must not be attributed to a particular CIC release without external provenance evidence.

## Schema

The file contains 57 numeric flow-summary features and two string target columns:

- `Label`: 33 fine-grained classes.
- `ClassLabel`: 8 broad classes.

No identifier, IP address, port, timestamp, raw packet payload, or source-dataset field is present. This makes the table suitable for a tabular flow classifier, but it prevents source-aware splitting or measuring performance by original dataset.

## Observed broad-label distribution

| ClassLabel | Rows |
| --- | ---: |
| Benign | 7,186,189 |
| DDoS | 1,234,729 |
| DoS | 397,344 |
| Botnet | 145,968 |
| Bruteforce | 103,244 |
| Infiltration | 94,857 |
| Webattack | 2,995 |
| Portscan | 2,255 |

Neither target column has missing values.

## Numeric-quality audit

The audit scanned every numeric value in 65,536-row batches using PyArrow and NumPy. It found no nulls, NaNs, infinities, or constant numeric columns. It did find values below zero in fields that are expected to represent durations, rates, counts, sizes, or time intervals.

| Feature | Negative values | Percent of rows |
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

The initial-window columns use exactly `-1` for every negative value, which is treated as an unavailable-value sentinel candidate. The header-length and minimum forward-segment-size anomalies are concentrated in DDoS rows. Across the non-window anomalies, 77,114 rows contain at least one negative value; 73,860 are DDoS. Removing all such rows would change class composition, so row deletion is not the Review-2 default.

## Review-2 preprocessing policy (proposed before implementation)

1. Use `ClassLabel` as the initial Review-2 target. It has eight broad classes and enough examples for a two-task demonstration. Preserve `Label` as fine-grained metadata for future experiments.
2. Exclude `Fwd Header Length`, `Bwd Header Length`, and `Fwd Seg Size Min` from the first model feature set because they contain systematic, class-concentrated invalid values. This leaves 54 candidate numeric features.
3. For remaining numeric features, replace negative values with missing values. In particular, map `-1` in both initial-window fields to missing values.
4. Fit any imputation and scaling on training data only, then apply the fitted transforms unchanged to validation and test data.
5. Convert model inputs to `float32`; preserve the exact retained feature order and class-to-ID mapping.

This policy is a Review-2 baseline. It has not yet been implemented, trained, or evaluated. The proposed decisions can be revised only with documented evidence and an updated data card.

## Sampling and split interface

Developer B implemented seeded, class-balanced reservoir sampling and 70/15/15 stratified splits in `src/data/sampling.py` and `src/data/splitting.py`.

The sampler scans `ClassLabel` in bounded batches and stores only original Parquet row indices. It does not materialize feature rows or write an output file. For the same file checksum, class order, limit, and seed, it selects the same rows. The splitter operates on those selected positions and guarantees that no selected position appears in more than one partition.

Observed smoke configuration:

| Setting | Value |
| --- | --- |
| Classes | Benign, DDoS, DoS, Botnet, Bruteforce, Infiltration, Webattack, Portscan |
| Per-class sample limit | 64 |
| Sampling seed | 42 |
| Split seed | 42 |
| Train/validation/test fractions | 70% / 15% / 15% |
| Selected rows | 512 |
| Train rows | 360 (45 per class) |
| Validation rows | 80 (10 per class) |
| Test rows | 72 (9 per class) |

The intended development configuration is 2,000 rows per class using the same class order and split fractions. It will produce 16,000 selected rows: 1,400 train, 300 validation, and 300 test rows per class. This configuration has not yet been materialized or passed through preprocessing.

## Preprocessing interface and observed smoke result

`src/data/preprocessing.py` implements the documented Review-2 baseline without modifying the raw Parquet file:

```text
selected original Parquet row indices
    -> materialize 54 retained features in requested row order
    -> negative or non-finite values become missing in memory
    -> fit per-feature median imputation on training rows only
    -> fit per-feature mean and standard deviation on imputed training rows only
    -> apply those fitted values to train, validation, and test rows
    -> output finite float32 matrices and stable ClassLabel IDs
```

The broad-class mapping is fixed in this order:

| ID | ClassLabel |
| ---: | --- |
| 0 | Benign |
| 1 | DDoS |
| 2 | DoS |
| 3 | Botnet |
| 4 | Bruteforce |
| 5 | Infiltration |
| 6 | Webattack |
| 7 | Portscan |

Observed real-file preprocessing smoke result used the 64-per-class sample and 70/15/15 split defined above:

| Partition | Feature matrix shape | Rows per class | Values finite? |
| --- | --- | ---: | --- |
| Train | `(360, 54)` | 45 | Yes |
| Validation | `(80, 54)` | 10 | Yes |
| Test | `(72, 54)` | 9 | Yes |

For the transformed training matrix, the greatest absolute per-feature mean was `5.5e-7`; feature standard deviations ranged from `0.9999972` to `1.0000035`. The small departure from exact `0` and `1` is expected after converting standardized arrays to `float32`.

No cleaned data file, cache, or model artifact was created. The intended 2,000-per-class development configuration has not yet been preprocessed.

## Next implementation decision

Materialize and validate the 2,000-per-class development configuration when needed. Define Task 1 and Task 2 only after agreeing with the model owner how a classifier adds new output classes and accepts optional replay data.
