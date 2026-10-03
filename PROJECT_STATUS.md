# Project Status

Last updated: 2026-10-03 (full clean result reviewed; full Task 1 prepared)

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

The one-epoch Kaggle T4 smoke, balanced-subset clean baseline, clean continual,
replay-label mitigation, balanced replay-exposure, and held-out novelty
comparisons have executed. Full-row preparation, throughput benchmarking,
and a full-partition clean static baseline also completed. The static result
is not deployment performance; full-data Task 1 has not run.

The static label-flip comparison has also executed on Kaggle. Reusable
task/replay, forgetting, frozen-teacher replay-label gate, optional balanced
training sampler, and novelty-evaluation APIs are implemented. New disk-backed
full-row preparation, durable training, and bounded static test evaluation
were exercised on Kaggle. A fresh five-class full Task 1 run is prepared.

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

## Historical novelty-baseline verification (2026-10-02)

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

- Confidence thresholding remains a simple baseline. The held-out Task 2
  evaluation ran and achieved low unknown recall (170/900); one class split
  and seed do not establish general novelty performance.
- PyTorch is unavailable in the local system environment; current sampler
  fit tests are skipped locally and included as Kaggle notebook checks.
  Earlier Kaggle T4 training and reload checks succeeded. The opt-in
  real-data local test remains skipped.
- Per-row dataset provenance is absent; source mixing cannot be evaluated from
  the Parquet file alone.
- Clean task/replay and six replay-label/gate conditions have run, but Benign
  results remain weak and filtering did not consistently improve accuracy.
  Balanced exposure and held-out novelty evaluation ran on a bounded subset;
  full-data continual Task 1/Task 2 training is pending and one-command
  integration remains incomplete.

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

Run `notebooks/kaggle_review2_full_task1_training.ipynb` on Kaggle T4 with
the existing prepared folder. Preserve its entire Task 1 output folder,
including checkpoints and clean replay buffer, for a separate Task 2 run.
Do not start Task 2 comparisons or tune against static test results here.

## Observed results

Small smoke, clean balanced-subset, static poisoning, clean continual,
replay-label mitigation, balanced replay exposure, and held-out confidence
novelty results were observed on bounded data. Full-row preparation and
full clean static model training completed; full-data continual training
has not executed.

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

## Observed Kaggle clean continual run reviewed on 2026-10-03

Inspected the untracked `data/continual/continual.ipynb`, `manifest.json`,
`summary.json`, `replay_buffer.npz`, and `training_curves.png`. The executed
notebook has no error outputs. Source revision:
`b2a0afc697272a9d093780057f38c63ee19340b0`; input checksum matches
the prior combined flow collection. Tesla T4 runtime changed to Python
`3.13.15`, PyTorch `2.11.0+cu128`, NumPy `2.1.3`, PyArrow `23.0.1`
(earlier clean/static runs: Python `3.12.13`, PyTorch `2.10.0+cu128`,
NumPy `2.0.2`, PyArrow `24.0.0` in the poisoning manifest). The 500-row
replay buffer is `float32` `(500,54)`, has unique original row IDs and 100
`int64` labels per old class 0–4. Task 1 checkpoint reload and both Task 2
arm reloads were reported identical.

Task 1 old-class test accuracy before Task 2 was
`0.9686666666666667`. Sequential Task 2 training left old accuracy `0`,
new accuracy `0.9377777777777778`, combined accuracy
`0.3516666666666667`, and Benign FPR `1.0`. With clean balanced replay,
old accuracy was `0.6826666666666666`, new accuracy
`0.9433333333333334`, combined accuracy `0.7804166666666666`, and
Benign FPR `0.9366666666666666`. Sequential/replay used 264/444 optimizer
steps (8/12 epochs; 4,200/4,700 training rows per epoch). The plot shows
the selected best validation epochs and strong divergence for sequential
training. Replay retains substantial old knowledge compared with sequential
fine-tuning, but old accuracy dropped by `0.286` from Task 1 and the Benign
false-positive rate remains severe. This balanced-subset, single-seed result
is not deployment-ready or an equal-compute comparison. The attached
checkpoint was not supplied; it is not recreated or claimed locally.

## Controlled replay-label poisoning and gate preparation on 2026-10-03

Fetched origin; local/remote unified branch matched
`a4676725d06aa993764b91f7f342d5a5b1f330ad` before edits. Published
the NumPy-only mitigation/audit source commit
`c636331fc9cdf2361357fd901d9c93411e3f59ba`, which the new notebook
pins. The notebook regenerates clean Task 1 in the active Kaggle runtime,
calibrates a 95th-percentile old-teacher label-inconsistency threshold on
clean Task 1 validation, and pairs clean/random-20%/targeted-20% replay
buffers with and without filtering. Random changes 100/500 labels;
targeted changes 80/400 eligible old-attack labels (16% of all replay).
The gate never receives simulator clean labels or changed indices; these
are only for post-decision audit. Task 2 rows and trusted validation remain
clean. No six-condition GPU run, mitigation metric, or replay-poisoning
outcome has been observed yet.
Focused `python3 -m unittest tests.test_label_consistency -v`: 5 passed.
Full `python3 -m unittest discover -s tests -p 'test_*.py' -q`: 31 passed,
1 opt-in real-data check skipped. Source `compileall`, notebook JSON/code
syntax/Markdown/empty-output checks (21 cells, 10 code), and
`git diff --check` passed. Local PyTorch/CUDA and the private Kaggle input
were unavailable; no Task 1 regeneration or six-condition GPU run occurred.

## Observed mitigation run and new phase prepared on 2026-10-03

Fetched origin and verified local/remote unified HEAD
`3b68e0347f4e9229a9a12143c4fc77d2519fcbb0` before edits. Reviewed
untracked `data/mitigation/manifest.json`, `comparison.json`,
`calibration.json`, executed `mitigation.ipynb`, and the training-curves plot.
The notebook had no error outputs. Its source was
`c636331fc9cdf2361357fd901d9c93411e3f59ba` on Tesla T4, Python
3.13.15, PyTorch 2.11.0+cu128, NumPy 2.1.3, and PyArrow 23.0.1. All six
initial expansion hashes matched and six checkpoint reloads passed.

Combined test accuracy was clean unfiltered/filtered `0.7804166667`/
`0.81375`, random20 unfiltered/filtered `0.81625`/`0.7858333333`, and
targeted20 unfiltered/filtered `0.81625`/`0.8279166667`. The gate rejected
100/100 random poisoned rows and 80/80 targeted poisoned rows, while falsely
rejecting about 4–5% of clean replay candidates. Benign FPR across conditions
remained `0.90`–`0.9533333333`. Success at identifying these labels did not
consistently improve classification: random20 filtered was worse than its
unfiltered pair. Optimizer steps differed (clean 444/444, random20 370/648,
targeted20 444/504, unfiltered/filtered). These are one-seed, unequal-work,
balanced-subset results; they do not prove broad poisoning resistance.

Published source commit `4e46f7f8ded77612210237e8dc9f6ea8e83730da`
adds optional `training_sampler='class_balanced'` with seeded inverse-class-
frequency replacement draws, while ordinary shuffled training remains the
default. The 500 unique stored replay examples are unchanged. Training
history now records expected and observed class exposure, unique replay rows
drawn, total draws, optimizer steps, and generator-state hashes by epoch.
Added known-validation fifth-percentile confidence calibration and external-
class-ID-aware novelty metrics, including defined empty-denominator behavior.
`notebooks/kaggle_review2_replay_balance_novelty.ipynb` pins that published
source, compares clean shuffled/balanced Task 2 arms with validation-only
strategy selection, and evaluates Task 1 known test against genuinely held-
out Task 2 test **before** expansion. It has no execution outputs or result
yet. Local full `unittest` discovery: 39 tests, 36 passed, 3 skipped (two
PyTorch-gated sampler tests and one opt-in real-data test). Source compilation
passed. Kaggle will run the torch-dependent synthetic checks; local PyTorch,
CUDA, and private input remain unavailable. The untracked attachments remain
outside Git. Streaming preparation/throughput benchmarking follows review of
the new Kaggle run.

## Kaggle replay-balance/novelty notebook stopped at test discovery

The first attempted Kaggle execution stopped in the notebook's environment/
synthetic-test cell with `ModuleNotFoundError` for
`tests.test_training_sampling` and `tests.test_novelty_evaluation`. Both test
files exist at the pinned source SHA, but `tests/` is not an importable
package (`__init__.py` is absent). No Task 1 training, replay-balance
comparison, or held-out novelty result is claimed from this stopped run.

Verified local/remote unified HEAD was
`0315bc041acec0a69629fda0764d43836dda576c` after fetching origin.
Changed only the notebook test invocation: two separate
`python -m unittest discover -s tests -p <exact filename> -v` subprocesses,
each with the repository as working directory, `check=True`, and an explicit
nonzero discovered-test count check. The pinned source SHA and all experiment
settings remain unchanged. Local exact-pattern discovery found 4 sampler
tests (2 passed, 2 PyTorch-dependent skipped) and 3 novelty-evaluation tests
(3 passed). Full local suite: 39 tests, 36 passed, 3 skipped (the two sampler
checks plus opt-in real-data integration). Notebook JSON, all 11 code-cell
syntax checks, Markdown staging, empty outputs, pinned SHA, and
`git diff --check` passed. Kaggle GPU/data execution remains unverified after
the correction; rerun the notebook from a fresh session before interpreting
any experiment result.

## Observed balanced replay/novelty run and full-row preparation phase

Fetched origin and verified local/remote unified HEAD
`8c09efccb5593247a8c57f34e200f262066e9400` before editing. Inspected
untracked `data/replay/replay.ipynb`, `manifest.json` (the supplied manifest
artifact), `test_summary.json`, `strategy_selection.json`, `novelty.json`,
and `training_exposure_novelty.png`. The executed notebook has no error
outputs. All seven targeted Kaggle tests passed (4 sampler, 3 novelty), and
Task 1 plus both Task 2 checkpoint reloads matched. Source
`4e46f7f8ded77612210237e8dc9f6ea8e83730da`; Tesla T4, Python 3.13.15,
PyTorch 2.11.0+cu128; input SHA-256
`666af53c788c79312b421c607289cb555c2335ae51ba39b2700a357595fa8dba`.

The **validation-only** criterion selected class-balanced replay: validation
macro-F1 `0.8529213667` versus shuffled `0.7541214312`. Subsequent test
combined accuracy was `0.8508333333` versus `0.7804166667`, old-class
accuracy `0.8353333333` versus `0.6826666667`, and new-class accuracy
`0.8766666667` versus `0.9433333333` (balanced versus shuffled). Benign FPR
improved but remained high at `0.6166666667` (shuffled `0.9366666667`).
Balanced/shuffled used 370/444 optimizer steps (10/12 epochs), so the result
is not equal-compute. One seed and a 2,000-per-class balanced subset do not
establish general or production performance.

The frozen fifth-percentile Task 1 validation confidence threshold was
`0.9247980118`. Held-out unknown recall was 170/900 = `0.1888888889`,
known false rejection 68/1500 = `0.0453333333`; unknown precision was
`0.7142857143`. This is an actual held-out-class result for the simple
confidence baseline, but poor unknown recall; it is not broad novelty
robustness. The plot shows near-uniform balanced draw shares, dominant
near-one confidence for both known and held-out rows, and the fixed cutoff.

Prepared `src/data/large_data.py`: full-row deterministic 70/15/15
class-stratified split, persisted original IDs/split indices/counts,
row-aligned bounded-batch float32 feature store excluding both targets,
and separate static versus Task-1-only continual preprocessing states.
Medians use a recorded seeded 200,000-row training-only reservoir per scope
(approximate); streaming population moments use all authorized training rows
after imputation. Completed stages can be reused; partial stages rebuild.
`src/training/disk_backed.py` supplies bounded-batch views and a trainer
compatible with existing model class IDs, save/load, validation-loss early
stopping, and deterministic shuffled/class-bucket balanced order without a
float64 weight per full-data row. Existing small-array APIs remain unchanged.

`notebooks/kaggle_review2_large_data_prepare.ipynb` pins published source
`5c0d519596bc6c4b99948126443635c2f7eb15f1`. It checks the reviewed
input hash, GPU and disk capacity, prepares all rows and both states, then
benchmarks one epoch on a fixed naturally distributed training-only subset
of up to 100,000 rows at batch sizes 128 and 256. It saves measured rows,
steps, timings, RAM/GPU peaks, storage costs, and labeled 1.5x-headroom
full-epoch estimates plus a provisional configuration. It does **not** run
full training, select by test, or produce large-data metrics. Local synthetic
suite: 4 tests, 3 passed, 1 PyTorch-dependent skipped; full suite 43 tests,
39 passed, 4 skipped. Source compilation, notebook JSON/code syntax and
empty-output checks, and `git diff --check` passed. GPU and full-row prep
are still unverified; next phase is real large-data training only after
reviewing this preparation run.

## Observed full-row preparation and full clean training notebook prepared

Fetched origin and verified local/remote unified HEAD
`96b8485b3d7f36fcb5d910fce6e5d4149de30a87` before editing. Inspected
untracked `data/preparation/preparation.ipynb` and **all** supplied JSON
artifacts: `manifest.json`, `split_manifest.json`, `features_manifest.json`,
`preprocessing_static.json`, `preprocessing_continual.json`,
`benchmark_results.json`, and `recommended_full_run.json`. The executed
notebook has eight executed code cells and no error outputs. Its source was
`5c0d519596bc6c4b99948126443635c2f7eb15f1`; Tesla T4, Python
3.13.15, PyTorch 2.11.0+cu128; input SHA-256
`666af53c788c79312b421c607289cb555c2335ae51ba39b2700a357595fa8dba`.

The **observed preparation** assigned all 9,167,581 original rows:
6,417,308 train, 1,375,139 validation, 1,375,134 test. It wrote one
54-feature float32 store (`1,980,197,624` bytes), separate static and
Task-1-only continual fit states (6,417,308 and 6,347,232 authorized
training rows), and about `2,146,034,791` bytes of persisted artifacts.
Both median reservoirs used 200,000 training rows, seed 42; medians remain
approximate. The attached JSON and notebook report split coverage and
complete stages, but the large `.npy` arrays were **not** attached locally
for an independent byte-level audit. Row-level splitting cannot rule out
duplicate-flow or capture-session leakage.

The **observed benchmark**, not full training, ran one epoch on 100,000
naturally distributed training rows. Batch 256 processed 391 optimizer steps
in `4.505302165` seconds, `22,196.0695` rows/s. Its `433.6787`-second
full-training-epoch projection includes a stated 1.5× headroom multiplier
and excludes the full validation pass; it is **not** a measured epoch.
Batch 128 measured `12,079.2591` rows/s. No full-data accuracy, validation,
or test metric exists yet.

Published source commit `91a26a1c13a563a7628e217f357dc27f73874784`
adds durable `train_full_disk_backed`: atomic per-epoch `latest.pt` plus
validation-loss-selected `best.pt`, full bounded validation, batch/epoch
progress and timing, and exact epoch-boundary restoration of model,
optimizer, scaler, Python/NumPy/Torch CPU/CUDA RNG, history, best state, and
patience. Resume rejects changed config, class IDs, data identity, or runtime.
An interrupted partial epoch repeats from the previous completed boundary;
mid-epoch resume is not claimed. A NumPy confusion accumulator supports
bounded full-test inference and accuracy, macro-F1, balanced accuracy,
per-class metrics, and Benign FPR without retaining all predictions.

`notebooks/kaggle_review2_full_clean_training.ipynb` pins that published
source, validates the prepared folder read-only (working or attached input),
then specifies a fresh eight-output hidden64/two-layer static model, natural
shuffled training over **all** 6,417,308 train rows, full validation-loss
selection, and a single complete test evaluation afterward. Config is
batch256, lr0.001, seed42, mixed precision off, max20 epochs, patience5.
It saves config/environment/data hashes, epoch checkpoints/history,
reload checks, curves, confusion matrix and metrics separately under
`/kaggle/working`. At this notebook-preparation step, it had no execution
outputs or full-training results; the subsequent run is recorded below.

Local focused streaming metrics: 2 passed; resume/header suite: 1 passed,
1 PyTorch-dependent resume-equivalence test skipped. Full `unittest` suite:
47 tests, 42 passed, 5 skipped. Source compilation, notebook JSON and code
syntax/empty-output checks, and `git diff --check` passed. Local PyTorch/GPU
and the 2.1-GB prepared arrays are unavailable; the Kaggle notebook reruns
the focused PyTorch check before training. Next work **after** inspecting the
actual clean full-data result is separate continual/replay and poisoning
comparison runs using the same prepared partitions.

## 2026-10-03: observed full clean static result and full Task 1 preparation

The untracked `data/full-train-v2/results/full_clean_20261003T045146_553311Z/`
manifest, training manifest, history, metrics, reload record, environment,
and plots were reviewed. They remain outside Git. A separately attached
`data/full-train.ipynb` contains three `NameError` outputs and does not
independently establish this successful run; the completed v2 result JSONs
are the evidence reported here. The
source was `91a26a1c13a563a7628e217f357dc27f73874784`, with the same
prepared-input identity as the earlier preparation run. On Kaggle Tesla T4,
Python 3.13.15 and PyTorch 2.11.0+cu128, the full static run completed
19 epochs and selected epoch 14 by full validation loss. Measured training
plus validation time was `104.2950168455` minutes. Full test accuracy was
`0.9828620338090688`, macro-F1 `0.757755762638612`, balanced accuracy
`0.7291289708762002`, and Benign FPR `0.00445669840657261` (4,804 of
1,077,928 Benign test rows). Infiltration recall was `0.059108799550182736`
and Webattack recall `0.12694877505567928`; 13,328 of 14,228 Infiltration
and 384 of 449 Webattack test rows were predicted Benign. The raw-count
plot is dominated by the majority class, so overall accuracy obscures
important minority weakness. The supplied reload JSON reports identical
bounded validation predictions and best weights; no independent local
checkpoint load was performed because PyTorch is absent locally. The
attached `.pt` files and prepared arrays were not committed.

Published `src/evaluation/plots.py` reads saved test confusion counts and
writes distinct raw-count, annotated row-normalized, and per-class recall/F1
plots without retraining or test inference. The existing raw-count plot is
preserved. `src/continual_learning/disk_replay.py` selects seeded,
training-only unique replay positions without materializing full features.
The new `notebooks/kaggle_review2_full_task1_training.ipynb` pins published
source `e6ebfcf7a9beec4d2b8fd5245e7c62314b9fadf9` and is **unexecuted**.
It will train a fresh five-output model on all 6,347,232 Task 1 training
rows using the verified continual-only fit state, full old-class validation,
durable resume, and bounded old-class test/held-out Task 2 novelty evaluation.
It will persist 100 unique old-class training exemplars per class for later
Task 2 use. No Task 2 model training or new novelty result is claimed.

Local synthetic plot tests: 3 passed; disk replay tests: 2 passed. Full
`python3 -m unittest discover -s tests -p 'test_*.py' -q`: 52 tests,
47 passed, 5 skipped (PyTorch-dependent or opt-in real data), zero failed.
`python3 -m compileall -q src tests`, 17-cell/8-code-cell notebook JSON,
syntax/Markdown/empty-output/source-pin checks, and `git diff --check`
passed. GPU training remains unavailable locally.
