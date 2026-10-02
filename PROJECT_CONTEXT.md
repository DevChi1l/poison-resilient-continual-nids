# Project Context

## Objective

Build a final-year B.Tech research prototype for poison-resilient, novelty-aware continual learning in a Transformer-based Network Intrusion Detection System (NIDS).

The research question is whether an adaptive NIDS can retain known attack knowledge, learn new attack classes, and resist corrupted update data without rejecting every unfamiliar sample.

## Review-2 scope

This review prioritizes a basic, working demonstration over a final research contribution. The minimum pipeline is:

```text
Parquet flow data -> preprocessing -> small tabular Transformer -> classification
         -> confidence novelty baseline -> Task 1 / Task 2 update
         -> label-flip poisoning comparison -> saved metrics
```

## Explicit baselines

- **Transformer:** a small tabular Transformer encoder is the classification backbone, not the novelty claim.
- **Novelty detection:** maximum-class-confidence threshold; this is a baseline only.
- **Continual learning:** sequential Task 1 to Task 2 training, with optional replay if stable in time.
- **Poisoning:** configurable label flipping of a bounded training fraction; it is not a claim to model every attacker.
- **Mitigation:** optional confidence- or anomaly-based filtering; do not delay the core pipeline to build it.

## Dataset decision

The Review-2 input is a local, untracked combined flow-level Parquet collection: `data/cic-collection.parquet`. It has 9,167,581 rows, 57 numeric flow features, a 33-class fine-grained `Label` target, and an 8-class broad `ClassLabel` target. The file does not contain per-row source-dataset provenance, so it must be described as a combined collection rather than attributed to CIC-IDS2017, CIC-IDS2018, or another source without external provenance evidence. Its audited schema, label counts, checksum, and data-quality findings are recorded in `data/metadata/combined_flow_collection_audit.md`.

For Review 2, use a documented, reproducible subset of this collection. Do not commit raw or processed data. Synthetic data may be used only for component smoke tests and must never be presented as NIDS evidence.

Synthetic data may be used only for a component smoke test and must never be presented as NIDS evidence.

## Environment observed on 2026-10-02

- System Python observed: 3.14.6.
- The current system environment used for the novelty task has NumPy 2.4.6,
  pandas, and PyArrow; the dependency manifest pins the data path separately.
- `torch` and `pytest` are not installed in the current system environment.
- Data/preprocessing, the tabular Transformer, NumPy novelty baseline,
  static label flips, clean two-task construction, replay selection,
  known-class/forgetting evaluators, and a simple replay-label consistency
  gate exist. Kaggle T4 smoke, balanced clean, static poisoning, and clean
  continual runs have been observed; see `PROJECT_STATUS.md`. None is a
  deployment or held-out unknown-attack result. Replay-label poisoning and
  the gate comparison are prepared but not yet run.

Dependency selection is pending. The first implementation owner must use versions compatible with the active Python runtime and record them in a shared dependency file.

## Research integrity

Use these terms accurately:

- **Hypothesis:** an expected outcome that has not been run.
- **Observed result:** metric produced by a recorded execution.
- **Baseline:** a comparison method, not a claimed contribution.
- **Not yet tested:** any unrun claim.
