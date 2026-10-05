# Project Status

Last updated: 2026-10-05 (clean Task2 acquisition notebook prepared; awaiting Kaggle run)

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
is not deployment performance. Full-data five-class Task 1 and the three-arm
full-data Task 2 comparison completed.

The static label-flip comparison has also executed on Kaggle. Reusable
task/replay, forgetting, frozen-teacher replay-label gate, optional balanced
training sampler, and novelty-evaluation APIs are implemented. New disk-backed
full-row preparation, durable training, and bounded static test evaluation
were exercised on Kaggle. The fresh five-class full Task 1 and three-condition
full Task 2 replay-label runs completed. A three-view local Streamlit demo now
provides real-artifact inference, persistent review, and saved-result playback.

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

- Confidence thresholding remains a simple baseline. The full Task 1 held-out
  Task 2 evaluation achieved low unknown recall (691/15,015); one class split
  and seed do not establish general novelty performance.
- CPU PyTorch 2.11.0 was installed only in the ignored local `.venv` for demo
  verification. Both real checkpoints loaded on CPU and produced normalized
  probabilities; no local GPU or training was used. The opt-in full Parquet
  fit test remains skipped because this session performed inference only.
- Per-row dataset provenance is absent; source mixing cannot be evaluated from
  the Parquet file alone.
- Clean task/replay and six replay-label/gate conditions have run, but Benign
  results remain weak and filtering did not consistently improve accuracy.
  Balanced exposure and held-out novelty evaluation ran on a bounded subset.
  The full Task 2 three-arm run is single-seed evidence and does not establish
  broad robustness. A one-command training pipeline remains incomplete.

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

Use the local artifact demonstration for Review 2 and independently review its
saved evidence. Do not launch more training from the UI. Any follow-up research
must predeclare additional seeds/threat models and preserve the current Task 2
archive as immutable evidence.

## Observed results

Small smoke, clean balanced-subset, static poisoning, clean continual,
replay-label mitigation, balanced replay exposure, and held-out confidence
novelty results were observed on bounded data. Full-row preparation and
full clean static model training and five-class Task 1 completed. The
three-condition full-data Task 2 experiment has executed and is recorded below.

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

Before any full Task 1 result was observed, the notebook's maximum epochs
was reduced from 20 to 12 solely for the Kaggle time budget. Validation-loss
selection, patience 5, the remaining configuration, durable per-epoch
latest/best checkpoints, strict resume, planned evaluation, and replay
export are unchanged. The completed eight-class static run remains unchanged.
The revised notebook's 17-cell/8-code-cell syntax, empty-output, source-pin,
resume/evaluation/replay and max12/patience5 checks passed. The 52-test
suite reported 47 passed and 5 existing environment/opt-in skips; source
compilation and `git diff --check` passed. No Task 1 GPU run was performed.

Local synthetic plot tests: 3 passed; disk replay tests: 2 passed. Full
`python3 -m unittest discover -s tests -p 'test_*.py' -q`: 52 tests,
47 passed, 5 skipped (PyTorch-dependent or opt-in real data), zero failed.
`python3 -m compileall -q src tests`, 17-cell/8-code-cell notebook JSON,
syntax/Markdown/empty-output/source-pin checks, and `git diff --check`
passed. GPU training remains unavailable locally.

## 2026-10-03: observed full Task 1 and prepared three-arm full Task 2

Fetched origin; local/remote unified branch initially matched
`c5ea706a12738c5b00836c200376e311d5fb3487`. Reviewed the attached
`data/full-run-task_01/results/full_task1_20261003T074354_763068Z/`
final/training manifests, history, environment, reload JSON, old-test
metrics, novelty metrics, replay manifest/buffer, and plots. The supplied
unexecuted notebook snapshot is not independent run evidence. Its recorded
source was `e6ebfcf7a9beec4d2b8fd5245e7c62314b9fadf9` on Kaggle Tesla
T4/Python 3.13.15/PyTorch 2.11.0+cu128. Full Task 1 used 6,347,232
old-class training rows, completed 12 epochs, selected validation-loss
epoch 9, and measured `65.50660256425` minutes of training plus validation.
On 1,360,119 old-class test rows, accuracy was `0.9925734439413022`,
macro-F1 `0.981162951652861`, balanced accuracy `0.9784715590433037`,
and Benign FPR `0.004171892742372403` (4,497/1,077,928).

The recorded held-out confidence threshold was the fifth percentile of
clean Task 1 validation maximum probabilities. It flagged only 691 of
15,015 unseen Task 2 test rows: unknown recall `0.04602064602064602`.
It falsely rejected 67,987 of 1,360,119 known test rows (rate
`0.04998606739557348`). This is weak novelty performance under this fixed
protocol, not a successful unknown-attack detector. The supplied replay
manifest reports 500 unique old training IDs, 100 per class. Locally, the
attached replay `.npz` independently matched its manifest IDs, shape,
labels/counts and finite float32 features; the attached `best.pt` byte hash
matched the reload report. PyTorch is absent locally, and the prepared
arrays were not attached, so we did **not** independently load weights or
verify replay rows against split codes/raw preprocessing here. The Task 2
notebook performs those checks against both full inputs on Kaggle.

Published backward-compatible source for a bounded Task 2 plus replay
sampler, optional per-epoch exposure in durable history, batched-score
consistency calibration, and count-based focused metrics at
`5546e7a4165eea51ba89e912603b5387e9506703`. New
focused old/new balanced-accuracy reporting was published at
`857dfbbcca2cfd6b33041ded51fbbae39b7a43f2`; the new
`notebooks/kaggle_review2_full_task2_targeted_mitigation.ipynb` pins this
latter source. It prepares exactly clean replay, targeted20 unfiltered, and the
**same** targeted20 buffer filtered. Targeted budget is 80/400 eligible
old-attack replay labels changed to Benign, or 16% of all 500 replay rows.
The frozen Task 1 teacher's 95th-percentile clean-validation gate applies
only to replay; simulator truth is reserved for separate audit. Each arm
draws 70,576 class-balanced examples per epoch from the same 70,076 clean
Task 2 training-row pool plus its replay pool. The full clean seen-class
validation loss selects each checkpoint; test metrics are later. Separate
arm folders hold durable latest/best/history, exposure, metrics and plots.
The notebook is **unexecuted**; no full Task 2 outcome is claimed.

Focused synthetic tests: Task 2 replay 2 passed, continual count metrics
2 passed, label consistency 6 passed. Full
`python3 -m unittest discover -s tests -p 'test_*.py' -q`: 57 tests,
52 passed, 5 skipped (local PyTorch or opt-in real data), zero failed.
`python3 -m compileall -q src tests`, notebook JSON/7-code-cell Python
syntax/15-cell Markdown/empty-output/source-pin checks, and
`git diff --check` passed. Local PyTorch/GPU and full prepared arrays are
absent, so GPU/data execution is not claimed.

## 2026-10-05: full Task 2 evidence reviewed and local demo completed

Fetched origin and verified `feature/unified-novelty` local/remote initially
matched `d3dc566`. The separate `origin/docs/faculty-presentation` branch
remained at `bc2dbbf`; no main merge or PPT/PDF edit occurred. Existing
untracked data/application artifacts were preserved. No training was launched.

Inspected `data/task2/results.zip` as data only. SHA-256 is
`0c73a44cca06cb92f7c21c98652808f739df67236d58d790f5cf4b93027d832e`.
ZIP integrity passed across 194 members; no absolute/traversal member,
symlink, or encrypted member was found. Named JSON evidence and numeric NPZ
metadata were read without importing or executing the bundled repository
snapshot or unpickling checkpoint files. The original ZIP was not modified.

The saved Task 1-before values are accuracy `0.9925734439413022`, macro-F1
`0.981162951652861`, and Benign FPR `0.004171892742372403`. Combined Task 2
accuracy for clean/poisoned/filtered was
`0.6010047020872148`/`0.44913586603196487`/`0.6077218656509111`; macro-F1
was `0.5664584687018178`/`0.5379282266745713`/`0.5578505675817005`; Benign
FPR was `0.4989118011592611`/`0.6885997951625712`/`0.4666276411782605`.
The gate rejected 80/80 poisoned and 42/420 clean candidates, retaining 378.
All arms completed six epochs, selected epoch one, took 1,656 optimizer
steps, and shared expanded-initial SHA-256
`e53fb3c899176c99cb4914a8dd2f03f3addba355e4b3566a5dd17df23853640a`.
Filtering improved aggregate accuracy and Benign FPR over poisoning but cut
DoS recall from `0.9605879095988322` to `0.6072884683142901` and increased
old-attack-to-Benign errors from 3,641 to 6,856. This single-seed evidence
does not establish broad poisoning resilience.

Implemented the local, three-view `streamlit_app.py` and `src/demo/` adapter
layer. Flow prediction enforces configured checkpoint/preprocessor hashes,
scope and exact feature order, defaults to the eight-class clean static model,
caps CSV/Parquet inference at 2,000 rows, uses CPU, and optionally reports
labelled metrics. Static and continual fit states cannot be interchanged; the
weak Task 1 max-confidence novelty baseline is restricted to the five-class
teacher. The quarantine view reproduces deterministic targeted corruption and
the existing teacher gate, stores only operational fields in an ignored SQLite
review queue, and records audited reject/release transitions without training.
Ground-truth attack masks appear only in an experiment-audit panel. The
continual view reads saved evidence only and displays Task 1-before, three-arm
metrics, forgetting, per-class values, normalized confusion matrices, and gate
trade-offs.

Configuration/setup is documented in `docs/LOCAL_DEMO.md`; the example is
`configs/demo_artifacts.example.json`; the short non-UI teacher command is
`scripts/demo_teacher_quarantine.py`. The evidence-linked compact record is
`docs/evidence/task2_targeted_summary.json`. Checkpoints, replay/results ZIP,
uploaded flows, and `.local/` SQLite state remain ignored and outside Git.

Focused demo tests passed 10/10 in the ignored local `.venv`, including all
three Streamlit views and the real archive regression. The complete unittest
suite ran 67 tests: 66 passed and the explicitly opt-in real-data *training*
smoke was skipped. Scoped `pytest -q` ran 70 tests: 70 passed, one opt-in test
skipped, with 65 subtests passed; `pytest.ini` prevents collection from private
archived repository snapshots under ignored `data/`. PyTorch 2.11.0+cpu loaded the real eight-class static
checkpoint and performed a normalized `(1, 8)` forward pass on one actual
bounded Parquet row with its static preprocessor. The real five-class Task 1
teacher performed a normalized `(500, 5)` forward pass over the supplied
replay buffer with its continual state; the live gate exactly reproduced 80
poison rejects, 42 clean rejects and 378 retained. The CLI demonstration
persisted 122 suspicious rows to a temporary SQLite database and printed its
separate experiment audit. The application and CLI never called `fit`; the
test suites exercised only existing tiny synthetic CPU fit tests. No research-
dataset training or new experiment ran.

## 2026-10-05: final local demo rehearsal and held-out sample export

Started from verified local/remote unified commit
`309708251a6a0c24ac04646dea0eabc057cd81ee`. Preserved the existing app,
all private artifacts, untracked `codexmem.tx`, and
`origin/docs/faculty-presentation` at `bc2dbbf`. No training, main merge,
PPT/PDF edit, checkpoint mutation, or result-archive mutation occurred.

The returned artifact bundle lacked its generated full split arrays. Re-ran
the repository's original labels-only `prepare_full_partitions` algorithm over
the SHA-verified Parquet with seed 42 into ignored `.local/` storage. All class
counts matched the saved manifest; reconstructed `splits.npy` SHA-256
`445182b087636e68089a5efd5d9645a1cc41109940d17ee778b81c935f0d2ca3`
exactly matched the training run record. The persisted held-out row-index hash
is `89f240936b161aec69f8ece5cf84d851647c953f68feb2d6a8daeae219e5d9a8`.

Added `scripts/export_demo_sample.py` and `src/demo/sample_export.py`. The
exporter hash-checks the original Parquet/split artifacts, scans only saved
test rows, uses seeded per-class reservoir selection without model predictions,
and writes 54 raw static feature columns in exact order plus `ClassLabel`.
Original row IDs are written to a separate provenance CSV. The real export at
`uploads/rehearsal/heldout_flows.csv` contains five examples from each of all
eight classes (40 rows); upload, provenance, manifest and split arrays remain
ignored. Its real CPU UI metrics were accuracy `0.725`, macro-F1
`0.7139496468443838`, and Benign FPR `0.0`; these are explicitly only bounded
rehearsal-sample behavior, not full-test performance.

Added a read-only `scripts/demo_preflight.py`/`src/demo/preflight.py`. It
checks hashes, model/preprocessor scope-width-head compatibility, Task 1
novelty identity, the 500x54 replay contract, safe readable Task 2 evidence,
Parquet/test-split alignment, and the SQLite destination, with concrete fixes.
All checks passed against the real artifacts. Added replay/novelty/dataset
hashes to the example config and a Streamlit `--database` override so rehearsal
uses an isolated queue.

Rehearsed all three UI views in a real local browser. The clean static model
predicted the 40 exported rows on CPU; the Task 1 teacher/gate added exactly
122 suspicious rows (80/80 poisoned plus 42/420 clean rejected; 378 retained)
to `.local/nids-demo-rehearsal-20261005/quarantine.sqlite3`. One reject and one
release were recorded with audit histories. After stopping and restarting
Streamlit, the UI still showed 120 pending, one rejected and one released row,
including both reasons. The default queue remained at zero rows. Release did
not trigger training. The saved continual view displayed all three exact arms,
normalized confusion/per-class data and the mixed-mitigation warning.

The compact record is `docs/evidence/demo_rehearsal_20261005.json`; exact setup
and timed narration are in `docs/LOCAL_DEMO.md` and
`docs/FIVE_MINUTE_DEMO.md`. Focused exporter tests passed 2/2 and existing demo
artifact tests passed 4/4. Full unittest discovery ran 69 tests: 68 passed and
the opt-in real-data training smoke was skipped. `.venv/bin/python -m pytest
-q` passed 72 tests plus 65 subtests with the same one skip. Compileall, JSON
parsing and `git diff --check` passed. The direct `.venv/bin/pytest` wrapper
does not add the repository root to `sys.path` in this local environment, so
use the verified module-form command above.

## 2026-10-05: clean Task 2 retention study prepared for Kaggle

Started from the user-verified unified commit
`9da2f2818b772d49f5fb44c4b205d4bff88df4ea`; local and remote matched after
fetch. Preserved the working demo, all original result artifacts, untracked
`codexmem.tx`, and `origin/docs/faculty-presentation` at `bc2dbbf`. No local
training, new test inference, main merge, app change, or PPT/PDF edit occurred.

Inspected the current Task 2 implementation. The existing view samples
uniformly across eight supplied classes, the verified Task 1 replay has 100
distinct rows per old class, checkpoint expansion preserves old head rows and
seeds the new head, Task 2 AdamW inherits model learning rate 0.001, and the
historical trainer selected by validation loss. These are documented as
hypotheses motivating an exploratory follow-up, not established causes of the
historical result.

Published backward-compatible source in `e3494d0` for optional exact class
quotas and clean-validation macro-F1 checkpoint selection. Published the
GPU-only auditable runner, finalized at
`57009efe09f48ee742576bcc5d089a71c30b4f0e`. It compares only A (existing
500-row replay/current sampler, lr 0.0001) and B (5,000 distinct training-only
rows per old class plus 62.5% original-proportion old exposure/37.5% uniform
new exposure, lr 0.0001). B is explicitly a combined intervention. Both retain
architecture, expansion seed/state, batch 256, 70,576 draws, maximum eight
epochs and patience three. Durable latest/best/resume state selects by full
clean-validation macro-F1 and records exposure/unique-row counts.

Added `notebooks/kaggle_review2_task2_clean_retention.ipynb`, pinned to
`57009ef`. It validates both private inputs and split/preprocessing/replay
isolation, locks the validation-only configuration choice, and evaluates only
the selected checkpoint on fixed test rows. Planned output includes
old/new/combined metrics, per-class recall/F1, Benign FPR, forgetting,
normalized confusion matrices, training curves, checkpoint/hash provenance,
and an explicit true/false report for the 85% combined-accuracy target.
Historical Task 2 artifacts remain separate and unchanged.

The complete local unittest suite passed 71 tests with the single opt-in real-
data smoke skipped. Runner compilation/help and notebook JSON/Python syntax,
empty-output/source-pin checks passed; `git diff --check` passed. Full prepared
arrays and a Kaggle GPU are unavailable locally, so the new study remains
unexecuted. Earlier Tesla T4 timings imply at most about 5.7 minutes for 16
train-plus-validation epochs and about 18.4 seconds for selected test
inference; budget 8–12 minutes overall for input hashing/preparation/plots and
runtime overhead.

## 2026-10-05: clean-retention evidence recorded; frozen-B poisoning notebook ready

Started from the user-verified unified commit
`99391300fa960e51e7bc9e021e4881e99b917af3`; local and remote matched after
fetch. Preserved the working demo, all original/private results, untracked
`codexmem.tx`, and `origin/docs/faculty-presentation` at `bc2dbbf`. No local
research training, app/PPT/PDF change, main merge, or artifact mutation
occurred.

Inspected `data/task2_clean_ret/results.zip` as evidence only. Its SHA-256 is
`1efb433f2f77501a290600a67a7f4e0b1f760b0f5b42b983db49ed1ffd2cacf0`;
ZIP integrity passed for 212 members with no unsafe path, symlink, encryption,
or extraction. Named JSON was read without executing archived source or
unpickling checkpoints. The attached notebook SHA-256 is
`daed240ae5420b66a5aa8fcb22b927694cdb25e46df084f3264654d1cad7f6a1`
and is byte-identical to the tracked launcher, but has zero executed code
cells and zero saved outputs; it is protocol evidence, while the ZIP JSON is
execution evidence.

Configuration B was selected. Fixed-test combined accuracy is
`0.9769527915097729`, macro-F1 `0.670591558187877`, Benign FPR
`0.010824470651101`, old accuracy `0.9874783015309689`, and new accuracy
`0.02350982350982351`; Infiltration recall is zero. Both A and B completed
four epochs, selected epoch one, and used 1,104 optimizer steps. This is
improved retention with weak acquisition, not successful eight-class
continual learning. The evidence-linked record is
`docs/evidence/task2_clean_retention_summary.json`.

Published the frozen three-arm source runner in `e543dfe` and its dependency-
light Kaggle checks in `489c27f85460b193a8ac8b9aa25d9e706f004b97`.
`scripts/run_task2_b_poisoning.py` reconstructs and hash-verifies the exact
25,000-row training-only replay, writes a private numeric NPZ, targets exactly
4,000/20,000 eligible attack rows to Benign, reuses the verified Task1 clean-
validation gate calibration, and stops if filtering empties an old supplied-
label bucket. Clean, poisoned, and filtered arms share fresh Task1 expansion,
configuration-B quotas, learning rate 0.0001, seed 42, batch 256, 70,576
draws/epoch, max eight epochs, patience three, and validation-macro-F1
selection. Actual epoch/step/exposure counts are saved. Test views open only
after all validation-selected checkpoint hashes are locked.

Added `notebooks/kaggle_review2_task2_b_poisoning.ipynb`, pinned to full source
SHA `489c27f85460b193a8ac8b9aa25d9e706f004b97`. It requires the complete
prepared folder, complete Task1 folder, clean-retention ZIP, and prior Task2
ZIP as private Kaggle inputs. No already-trained B checkpoint initializes an
arm. It preserves all earlier studies separately and writes checkpoints,
private replay/audit NPZs, metrics, normalized confusion/per-class plots,
training curves, provenance, clean-control reproducibility, and gate tradeoffs
under `/kaggle/working/task2_b_poisoning_study`.

Full local `pytest` passed 77 tests plus 65 subtests, with one explicitly
opt-in real-data training smoke skipped. The 23-test focused selection plus 30
subtests covered archive safety, exact attack budget, replay isolation, fixed
quotas, filtering failure, streaming metrics and durable checkpoint/resume;
the new dependency-light protocol suite also passed 3/3 under `unittest`.
Both evidence bundles passed hash/JSON checks. The notebook parses as 10
cells/4 syntactically valid code cells with empty outputs and the exact source
pin; compileall, runner help, remote pin and `git diff --check` passed.
GPU/data research training remains intentionally unexecuted.
The historical B run measured 92.951 seconds for four train-plus-validation
epochs (23.238 seconds/epoch); the three-arm 24-epoch cap is therefore about
9.30 GPU minutes. Historical fixed old+new test inference was 19.067 seconds
per arm, or about 0.95 minutes for three. Budget 13–18 minutes overall for
hashing, replay/gate preparation, plots and Kaggle overhead.

## 2026-10-05: frozen-B Kaggle evidence folder compatibility

Verified local and remote `feature/unified-novelty` at
`7cc01bd8d3e0034cc9cb99911ccfe79165968179` before editing. Preserved the
private archives, demo, faculty branch, and untracked `codexmem.tx`.

Published runner and test changes first at
`1c2b7ad76a4c7078e43ae867fa86136b997c6780`. The runner now accepts an
original ZIP or an extracted study folder for each of the retention and prior
Task2 inputs. ZIP inputs still require their original full SHA-256. Folder
inputs verify the exact nine JSON files consumed: eight retention records and
the prior calibration record. Their byte hashes were derived directly from
the original locally verified ZIPs and are pinned in the runner. Semantic,
checkpoint, replay, split, attack, and gate checks remain in place. No
experiment setting changed.

The Kaggle notebook now pins `1c2b7ad` and provides explicit folder selectors
for the supplied `results_ret/task2_clean_retention_study` and
`results/full_task2_targeted_20261003T095508_992113Z` paths. Automatic
discovery accepts exactly one hash-valid ZIP or folder per study and gives an
ambiguity/error message otherwise. The original ZIP option remains available.

Verification: four focused protocol tests passed, including ZIP/folder
equivalence and altered-file/symlink rejection. A separate read-only check
reconstructed only the consumed JSON files in temporary folders from both
original ZIPs; ZIP, folder, and mixed inputs yielded identical evidence and
passed the existing semantic checks. Altering one extracted JSON file was
rejected. The notebook selector accepted one valid folder, rejected an
ambiguous ZIP+folder pair, and rejected altered evidence. Notebook JSON and
four code-cell syntax checks passed. No training was launched.

## 2026-10-05: clean Task2 acquisition study awaiting Kaggle execution

Started from matching local/remote unified HEAD `129c007face039a985e1abce51a70e431731485e`.
The uploaded frozen-B run's clean arm exactly reproduced prior B: combined
accuracy `0.9769527915097729`, macro-F1 `0.670591558187877`, old accuracy
`0.9874783015309689`, new accuracy `0.02350982350982351`, and zero
Infiltration recall. This is inadequate eight-class acquisition. The
unchanged quarantine gate rejected 3,991/4,000 poison and 2,165/21,000 clean
training replay rows; nine poison rows remained. It is not network blocking.

No blocking implementation bug was confirmed after class/feature/label,
checkpoint/head, gradient, exposure, preprocessing and selection audits.
Published the source runner and old-only optional distillation at
`d8711cf5e35b23951ad705964581ae4ef7a58fc7`; full local suite: 82 passed,
one opt-in skipped, 65 subtests. The new one-notebook two-clean-arm protocol
is pinned to that SHA. A uses revised exposure/coverage and CE; B is identical
except frozen Task1 teacher KD on old replay only. Max 12 epochs/arm, patience
four, validation-only selection. No research-data training occurred locally.
Details and measured-runtime extrapolation are in the final walkthrough section.

Next: run `notebooks/kaggle_review2_task2_clean_acquisition.ipynb` on Kaggle,
preserve its complete private output, then review every class's recall/F1,
new/old accuracy and forgetting before any app integration. Pending final
integration: latest/historical playback, correctly paired inference
checkpoints, live quarantine evidence, preflight, findings Markdown and
teacher script. Working demo, original results, faculty branch and PPT/PDF
are unchanged. Untracked `codexmem.tx` remains untouched.
