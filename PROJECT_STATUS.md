# Project Status

Last updated: 2026-10-02

## Current state

Developers A+B+C are one owner of model, data, continual learning, novelty,
poisoning, mitigation, evaluation, integration, and shared documentation.

The bounded Parquet data path, deterministic sampling/splitting, train-only
preprocessing, tabular Transformer classifier source, and NumPy-only confidence
novelty baseline are implemented. The active branch is
`feature/unified-novelty`, created from verified commit `93492d6` on
`origin/feature/model-nids`. No merge to `main` and no push occurred.

There is no executable end-to-end pipeline, checkpoint, or model/novelty
experiment result. Real-data model training has not been verified.

## What is confirmed

- The local combined flow collection remains ignored and untracked. Its
  per-row source-dataset provenance is unavailable.
- `src.data` produces finite `float32` matrices with 54 retained features and
  `int64` broad-class labels, with preprocessing fitted on training rows only.
- `TabularTransformerClassifier.predict_proba()` output columns align with
  `model.class_ids`; a column position is not assumed to be an external ID.
- `src.novelty.detect_unknown(probabilities, threshold)` returns
  `max(probabilities, axis=1) < threshold` as a boolean mask. Equality is known.
- `src.novelty.summarize_predictions()` optionally returns external predicted
  class IDs, maximum probabilities, and unknown flags after validating an
  explicit `class_ids` mapping.
- Novelty validation rejects malformed, empty-column, non-finite, out-of-range,
  or non-normalized probability matrices and invalid thresholds. Zero-row
  batches are supported, and inputs are not mutated.
- The unrelated untracked `data/data_vis.ipynb` was preserved unchanged.

## Verification observed in this task

- `python3 -m unittest tests.test_novelty -v`: 9 passed, 0 failed, 0 skipped.
- `python3 -m unittest discover -s tests -p 'test_*.py' -v`: 16 passed,
  0 failed, 1 skipped. The skipped check was the opt-in real-data
  model/data training smoke test.
- `python3 -m compileall -q src/novelty tests/test_novelty.py`: passed.
- `git diff --check`: passed on the complete code and documentation diff.
- `python3 -m pytest -q`: did not start because `pytest` is not installed.
- Current system checks found NumPy, pandas, and PyArrow available; PyTorch and
  pytest are not installed. No packages were installed for this task.

## Known blockers and limitations

- Confidence thresholding is only a component baseline. Unknown-attack
  detection performance cannot be claimed until known classes are used for
  training and genuinely held-out classes are used as unknown evaluation data.
- No threshold has been calibrated on a validation protocol, and no novelty
  metric has been produced.
- PyTorch is unavailable in the current system environment, so real-data model
  training and the opt-in model/data integration test remain unverified.
- Per-row dataset provenance is absent; source mixing cannot be evaluated from
  the Parquet file alone.
- Continual tasks, replay, label-flip poisoning, mitigation, evaluation, and
  one-command integration are not implemented.

## Immediate next action

Implement deterministic, configurable training-label flipping while preserving
the original labels and recording changed indices, requested/achieved rates,
and seed. After that, continual tasks, replay, mitigation, evaluation, and
integration remain the unified team's responsibility.

## Observed results

Only component test outcomes are reported above. No accuracy, F1, novelty,
poisoning, forgetting, or mitigation result has been generated.
