# Project Status

Last updated: 2026-10-03 (clean continual/replay preparation)

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

There is no executable end-to-end continual pipeline. The one-epoch Kaggle T4
smoke and a later balanced-subset clean-baseline run completed. Neither
establishes production/full-dataset or unknown-attack performance.

`notebooks/kaggle_review2_clean_baseline.ipynb` and the static label-flip
comparison have executed on Kaggle. Reusable clean two-task construction,
class-balanced replay selection, and forgetting metrics are implemented.
`notebooks/kaggle_review2_continual_replay.ipynb` is prepared but has not
been executed. Replay poisoning and mitigation are not implemented.

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
- Clean task/replay preparation is implemented but its Kaggle training
  comparison has not run. Replay poisoning, mitigation, held-out novelty
  evaluation, and one-command integration are not implemented.

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

Run the clean continual/replay notebook on Kaggle T4 with the same attached
input. Review Task 1 and both Task 2 arms' histories, metrics, forgetting,
training-work counts, row indices, plots, and reload checks before choosing
the next implementation task.

## Observed results

The small smoke, clean balanced-subset, and static poisoning comparison
results were observed. Continual forgetting, replay poisoning, held-out
novelty, and mitigation have no observed result yet.

## Observed Kaggle clean baseline reviewed on 2026-10-03

The attached untracked `data/manifest (1).json`, `data/metrics (1).json`,
`data/training_history (1).json`, executed `data/baseline-lite.ipynb`, and
training/confusion plots were inspected and preserved outside Git. The source
was `68135601a3ddd95832c57bb602f210f1a0e18a43`; the input checksum was
`666af53c788c79312b421c607289cb555c2335ae51ba39b2700a357595fa8dba`.
On Tesla T4 (Python 3.12.13, PyTorch 2.10.0+cu128), the 2,000-per-class
sample produced 11,200/2,400/2,400 train/validation/test rows, each with 54
features. Training completed 20 epochs; the best validation-loss epoch was
17. Test accuracy was `0.8833333333333333`, macro-F1
`0.878016638838182`, training time `13.00852884599999` seconds, and checkpoint
reload predictions were identical. Benign recall was `0.44`: 168/300 Benign
test rows were predicted as attacks (56% false-positive rate), including 138
predicted as Infiltration. This is a balanced-subset baseline, not production
or full-dataset performance. The clean notebook remains unchanged.

## Static label-flip preparation on 2026-10-03

Fetched origin; local/remote unified branch were both `5e07e3ec681e000a469d23ecbf1e658697e7a815`
before edits. The new API/test commit was normally pushed at
`316b817b40f4008af128489ac958d3b232b56049`, and the new notebook pins
that published SHA. `src/poisoning` supports deterministic random and targeted
training-label flips with explicit different budget denominators. The
NumPy-only `src/evaluation` summarizes known-class classification and attack
diagnostics. The notebook prepares seven conditions (0% control, three random,
three targeted) using identical splits/preprocessing and clean validation.
No Kaggle poisoning condition has been executed yet. Local focused tests:
6 passed. Full `unittest` discovery: 22 passed, 1 opt-in real-data model test
skipped. `compileall`, notebook JSON/code syntax and Markdown-stage checks,
and `git diff --check` passed. The notebook has 15 cells (7 code), all with
empty outputs. Local PyTorch/GPU and the private dataset remain unavailable.

## Observed Kaggle static label-flip comparison reviewed on 2026-10-03

The attached, untracked `data/poison/manifest.json`, `comparison.json`, and
executed `poison.ipynb` were inspected. The notebook had no error outputs.
Source SHA: `316b817b40f4008af128489ac958d3b232b56049`; input SHA-256:
`666af53c788c79312b421c607289cb555c2335ae51ba39b2700a357595fa8dba`.
All seven conditions report identical initial-weight hashes and successful
checkpoint reload checks on Tesla T4. The 0% control repeated accuracy
`0.8833333333333333`, macro-F1 `0.878016638838182`, and Benign false-positive
rate `0.56`.

At random 20%, 2,240 of 11,200 training labels changed: accuracy `0.87`,
macro-F1 `0.8590190796024948`, Benign FPR `0.67`. At targeted 20%, 1,960
of 9,800 eligible attack-source labels changed (17.5% of all 11,200 training
rows): accuracy `0.8533333333333334`, macro-F1 `0.8514854740614037`, and
true-attack-to-Benign error rate `0.06571428571428571` versus control
`0.025238095238095237`. Random rates divide by all rows; targeted rates
divide by eligible source rows. They are not equal budgets. Effects are mixed:
random 10% accuracy was `0.88375`, slightly above control, and targeted 10%
macro-F1 was `0.8817804870790201`, also above control. One seed and one
balanced subset do not support a universal degradation claim or production
estimate. No attached artifacts were committed.

## Clean continual/replay preparation on 2026-10-03

Fetched origin: local and remote unified branch matched
`a42f464ad1081e7bab21ff11db77992df0a347b9` before edits. Published
NumPy-only task/replay/evaluation source commits
`ca445b425ef96ad4b0d8e214604dec8e49fb0916` and
`b2a0afc697272a9d093780057f38c63ee19340b0`, the latter of which the
new notebook checks out. Task 1 uses global IDs 0–4, Task 2 new IDs 5–7. Task construction
uses the existing selected-row loader, label encoder, and preprocessor helper;
only Task 1 training rows fit preprocessing. The replay selector takes 100
unique Task 1 training rows per old class with seed 42, excluding validation
and test row IDs. Forgetting keeps the same five-class F1 average and signed
before-minus-after differences. The notebook compares Task 2 sequential
fine-tuning against replay from independent reloads of one Task 1 checkpoint.
It records training-work differences. No continual GPU run has occurred.
Focused `python3 -m unittest tests.test_continual_learning -v`: 4 passed.
Full `python3 -m unittest discover -s tests -p 'test_*.py' -q`: 26 passed,
1 opt-in real-data test skipped. Source `compileall`, notebook JSON/code
syntax/Markdown/empty-output checks (23 cells, 11 code), and
`git diff --check` passed. Local PyTorch/CUDA and private dataset execution
were unavailable; no GPU training or continual result was generated here.
