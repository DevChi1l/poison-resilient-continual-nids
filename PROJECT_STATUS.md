# Project Status

Last updated: 2026-10-02

## Current state

**Developer A Model/NIDS implementation is ready on `feature/model-nids`; integration has not started.**

The repository contains the project blueprint, research report, GitHub templates, persistent Review-2 coordination documents, and the three remote feature branches. The Developer A branch now contains a tabular Transformer model, training/inference/checkpoint API, and torch-gated smoke test. No data pipeline, dependency manifest, integrated executable pipeline, checkpoint, or experiment result exists.

## What is confirmed

- Repository default branch: `main`
- Review-2 branches created: `feature/model-nids`, `feature/data-continual`, `feature/novelty-poisoning`
- Python runtime: 3.14.6
- Locally available: `numpy`, `pandas`
- Locally absent: `torch`, `scikit-learn`
- Dataset files are not tracked in Git. Local candidate artifacts exist under `data/` and are ignored: `archive.zip` (605 MiB), `archive (1).zip` (825 MiB), and `cic-collection.parquet` (979 MiB).
- Archive contents are named as CSE-CIC-IDS2018-style Parquet files, not CIC-IDS2021. Developer B must verify source provenance before assigning the primary-dataset label.
- A safe ZIP-only extraction utility and its smoke tests are ready on `feature/model-nids`; it does not preprocess data.

## Immediate next actions

1. Assign teammates to the three existing feature branches in `TEAM_WORKFLOW.md`.
2. Developer B verifies the local archive provenance, selects the documented primary dataset, and consumes the extracted Parquet files through the data/task interface.
3. Developer B coordinates a compatible shared PyTorch dependency manifest; then Developer A runs the torch smoke test locally or in a GPU environment.
4. Developer C implements standalone label-flip, confidence novelty, and metric smoke tests against documented inputs.

## Known blockers

- No dataset subset has been added or documented.
- No ML framework is installed or pinned for the current Python version.
- Teammate GitHub usernames and branch assignments have not yet been recorded.
- The Model/NIDS branch cannot run its runtime smoke test locally until PyTorch and pytest are installed.
- The stated CIC-IDS2021 default conflicts with the names in the currently available ZIP archives; this must be resolved before experiments are labelled.

## Observed results

None. No experiment has been run.
