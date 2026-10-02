# Project Status

Last updated: 2026-10-03 00:02 IST

## Current state

Developers A+B+C are one owner of model, data, continual learning, novelty,
poisoning, mitigation, evaluation, integration, and shared documentation.

The bounded Parquet data path, deterministic sampling/splitting, train-only
preprocessing, tabular Transformer classifier source, and NumPy-only confidence
novelty baseline are implemented. The active branch is
`feature/unified-novelty`, created from verified commit `93492d6` on
`origin/feature/model-nids`. The completed feature commit `8070927` was
published to `origin/feature/unified-novelty` with upstream tracking on
2026-10-02. No merge to `main` occurred.

There is no executable end-to-end pipeline. A real-data, one-epoch Kaggle T4
smoke run completed and its attached manifest, metrics, history, and executed
notebook were reviewed. This small run is preliminary; it does not establish
clean-baseline quality or unknown-attack detection performance.

`notebooks/kaggle_review2_clean_baseline.ipynb` is prepared for a fresh
2,000-per-broad-class training run, with up to 20 epochs and validation-loss
early stopping. It has not been executed on Kaggle. Its source pin is the
published progress-reporting API commit
`68135601a3ddd95832c57bb602f210f1a0e18a43`.

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

## Publication verification on 2026-10-02

- `git fetch --prune origin` confirmed the remote branches before publication.
- The local branch pointed to `8070927`; it had no tracked changes. The only
  untracked path was `data/data_vis.ipynb`, which was preserved.
- The commits introduced relative to `origin/main` contained code, tests, and
  documentation, with no dataset, archive, checkpoint, secret, or generated
  result file.
- Fresh checks: focused novelty suite 9 passed; repository `unittest` suite
  16 passed and 1 opt-in real-data model test skipped; compilation and
  `git diff --check HEAD^ HEAD` passed.
- `git push -u origin feature/unified-novelty` succeeded. A subsequent
  `git ls-remote --heads origin feature/unified-novelty` returned the same
  full SHA as local `HEAD` at that point:
  `8070927eddac8bef6e940cd7d2f8c5eda3e9dee6`.

## Known blockers and limitations

- Confidence thresholding is only a component baseline. Unknown-attack
  detection performance cannot be claimed until known classes are used for
  training and genuinely held-out classes are used as unknown evaluation data.
- No threshold has been calibrated on a validation protocol, and no novelty
  metric has been produced.
- PyTorch is unavailable in the local system environment, so the model's new
  optional progress path could not be run locally. The earlier Kaggle T4 smoke
  training and checkpoint reload did succeed. The local opt-in real-data test
  remains skipped.
- Per-row dataset provenance is absent; source mixing cannot be evaluated from
  the Parquet file alone.
- Continual tasks, replay, label-flip poisoning, mitigation, evaluation, and
  one-command integration are not implemented.
- The clean baseline notebook has passed local structure and syntax checks
  only. Its larger GPU fit, metrics, checkpoint reload, and saved manifest
  still require a recorded Kaggle execution.

## Kaggle notebook preparation verification on 2026-10-02

- Fetched `origin`; `origin/feature/unified-novelty` and local `HEAD` matched at
  `22e797660307bfed62b739d61b5e4c05bcb0a187` before notebook edits.
- Validated notebook JSON, code-cell Python syntax, Markdown before each code
  stage, and empty execution counts/outputs: 19 cells, 9 code cells, passed.
- `python3 -m unittest tests.test_novelty -q`: 9 passed.
- `python3 -m unittest discover -s tests -p 'test_*.py' -q`: 16 passed,
  1 opt-in real-data model test skipped.
- PyTorch and `nbformat` are unavailable locally. No local GPU training or
  real-data run was performed. The unrelated `data/data_vis.ipynb` remains
  untracked and untouched.

## Observed Kaggle smoke execution reviewed on 2026-10-03

The user supplied untracked `data/manifest.json`, `data/metrics.json`,
`data/training_history.json`, and `data/notebook-smoke.ipynb`. The JSON files
agree, and the executed notebook's code cells have no error outputs. These
artifacts remain local and uncommitted.

- Source revision: `22e797660307bfed62b739d61b5e4c05bcb0a187`.
- Input: combined flow collection with unverified source provenance; attached
  SHA-256 `666af53c788c79312b421c607289cb555c2335ae51ba39b2700a357595fa8dba`.
- Kaggle environment: Tesla T4, Python 3.12.13, PyTorch 2.10.0+cu128; CUDA
  was available and used.
- Seed 42 and 64 rows per broad class yielded train `(360, 54)`, validation
  `(80, 54)`, and test `(72, 54)`.
- One epoch: train loss `2.1014001899295383`, validation loss
  `1.9477939367294312`, validation accuracy `0.275`.
- Test accuracy `0.25`; macro-F1 `0.1915376915376915`. Checkpoint reload
  predictions were identical.
- At threshold `0.5`, all 72 known-class test rows were flagged unknown. This
  is a confidence diagnostic, not unknown-attack detection performance.
- The smoke manifest observed at least 2,255 source rows for every broad
  class, supporting the planned 2,000-per-class sample on the same file.

## Clean baseline preparation verification on 2026-10-03

- Fetched `origin`: local and remote unified branch matched at `2b2f5ae`
  before editing. `main` remained behind.
- Added backward-compatible `fit(..., verbose=True)` progress reporting,
  published as full SHA `68135601a3ddd95832c57bb602f210f1a0e18a43`.
  Default callers remain quiet, and validation-loss best-state logic is
  unchanged.
- `python3 -m compileall -q src/models tests/models`: passed.
- `python3 -m unittest discover -s tests -p 'test_*.py' -q`: 16 passed,
  1 opt-in real-data model test skipped. The PyTorch `pytest` model test could
  not run locally because PyTorch and pytest are absent.
- Clean notebook JSON and Python code syntax checks passed: 21 cells,
  10 code cells, no saved execution outputs. No clean baseline training result
  has been generated.

## Immediate next action

Run the clean baseline notebook on Kaggle T4 with the same attached input.
Review its manifest, history, per-class metrics, curves, confusion matrix, and
checkpoint reload result before choosing the next implementation prompt.
Deterministic training-label flipping and continual learning remain later
unified-team responsibilities.

## Observed results

Only the small observed Kaggle smoke metrics above exist. The new clean
baseline, novelty performance, poisoning, forgetting, and mitigation have no
observed result yet.
