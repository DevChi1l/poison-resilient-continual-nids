# Source code

## Current implementation

`src/data/` contains the first implemented Review-2 component, now owned with
all source packages by the unified A+B+C team:

- Parquet metadata inspection without loading all rows.
- Strict validation of the audited combined-flow schema.
- Stable, original-order selection of 54 retained numeric features.
- Bounded batch iteration over those features plus `Label` and `ClassLabel`.
- Seeded, class-balanced reservoir sampling that stores only raw-file row indices.
- Seeded 70/15/15 stratified train/validation/test splitting.
- Selected-row materialization in bounded batches, without rewriting the raw file.
- Train-only negative-to-missing conversion, median imputation, standardization, and stable broad-label encoding into `float32` partitions.

The data package does not yet cache processed samples or build continual tasks.
The model source and novelty baseline exist, but real-data training and the
end-to-end pipeline remain unverified.

Keep package boundaries aligned with the architecture in `docs/PROJECT_BLUEPRINT.md`.
