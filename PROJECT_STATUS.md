# Project Status

Last updated: 2026-10-02

## Current state

**Dataset audit, bounded loading, seeded sampling/splitting, and train-only preprocessing are complete.** A local Parquet collection is available and untracked. Its source-dataset provenance remains unavailable within the file.

The repository contains the project blueprint, research report, GitHub templates, persistent Review-2 coordination documents, and a local `data/cic-collection.parquet` (~979 MiB). The previously local-only data commit was verified absent from `origin/main`, then amended so the data file remains on disk but is no longer tracked. Its checksum is recorded in `data/metadata/combined_flow_collection_audit.md`. Developer B has added a shared direct dependency manifest, `src/data/` bounded schema loader, class-balanced sampling/splitting, and train-only preprocessing with synthetic/real smoke verification. There is no executable pipeline, model, checkpoint, or experiment result.

## What is confirmed

- Repository default branch: `main`
- Active Developer B branch: `feature/data-continual`.
- System Python: 3.14.6; Developer B `.venv`: Python 3.12 with `numpy`, `pandas`, and `pyarrow`.
- Local dataset: `data/cic-collection.parquet`, 9,167,581 rows, 59 columns, and 57 numeric flow features.
- Targets: `Label` has 33 classes; `ClassLabel` has 8 classes. Neither target has missing values.
- No numeric feature has null, NaN, infinity, or constant values, but several contain negative invalid/sentinel values. Audit counts and the proposed cleaning policy are documented in `data/metadata/combined_flow_collection_audit.md`.
- The Parquet file is ignored and untracked. Do not commit or push it.
- `src.data.inspect_parquet()` validates the strict 57-feature combined-flow schema from Parquet metadata; `iter_dataset_batches()` yields bounded batches of 54 retained features plus both targets.
- `sample_class_balanced_indices()` selects reproducible, class-balanced original row indices without materializing the dataset; `stratified_split_indices()` creates deterministic non-overlapping 70/15/15 partitions.
- `prepare_sampled_dataset()` materializes selected rows, maps negative/non-finite values to missing in memory, fits median imputation and standardization on train rows only, and returns finite `float32` matrices with stable broad-class IDs.
- Seven synthetic data tests passed. A real-data smoke run selected 64 rows from each of eight classes, produced 45/10/9 train/validation/test rows per class, and created finite `(360, 54)`, `(80, 54)`, and `(72, 54)` in-memory matrices.

## Immediate next actions

1. Run the configurable 2,000-per-class development path once its configuration/entry point is added; do not overwrite or commit the raw Parquet file.
2. Confirm the Model A class-ID, output-head expansion, and replay interfaces before implementing the Task 1/Task 2 runner.
3. Implement Task 1/Task 2 construction using fixed held-out evaluation partitions after the model interface is confirmed.
4. Extend the shared dependency manifest when Developers A and C add their tested components.

## Known blockers

- Per-row source-dataset provenance is absent; source mixing cannot be evaluated from the Parquet file alone.
- No ML framework is installed or pinned for the current Python version.
- Task construction and model integration are blocked on the Developer A interface; no model run has occurred.

## Observed results

No model experiment has been run. Dataset structure, label distribution, negative-value counts, and invalid-value overlap were observed in read-only audits. The data package passed seven synthetic tests, strict real-file schema/batch validation, and a real-data 64-per-class sampling/splitting/preprocessing smoke run.
