# Handoff

> Latest continuation state: see the final dated section, “Full clean result
> reviewed; full Task 1 prepared.” Earlier sections are historical and do not
> override current unified ownership or implementation status.

Date/time: 2026-10-02

Developer: Repository initialization agent

Branch: `main`

## Completed

- Inspected the remote repository, current branch, commits, documentation, dependency files, source files, tests, and local runtime.
- Confirmed that the repository is a blueprint only; no implementation was overwritten.
- Added the persistent context system and three-developer Review-2 interface contract.
- Reworked `README.md` into the living Review-2 project front door, including status, ownership, interfaces, run-state rules, and documentation-update protocol.

## Currently working

No active implementation task.

## Files changed in this handoff

- `AGENTS.md`
- `PROJECT_CONTEXT.md`
- `TEAM_WORKFLOW.md`
- `ARCHITECTURE.md`
- `PROJECT_STATUS.md`
- `HANDOFF.md`
- `REVIEW2_CHECKLIST.md`
- `README.md`
- `docs/TEAM_WORKFLOW.md`
- `README.md`
- `TEAM_WORKFLOW.md`

## Tests run

- Repository inspection and Git status/log checks.
- Python runtime package availability check.
- Markdown/trailing-whitespace validation passed with `git diff --check` before commit.

## Results

No experiment results. The repository has no dataset or implementation.

## Known problems

- Dataset and framework dependencies are absent.
- Three developers must be assigned to the documented branches before parallel coding starts.

## Important decisions

- Review 2 uses a small tabular Transformer, confidence-threshold novelty baseline, Task 1 to Task 2 continual-learning demonstration, and configurable label-flip poisoning.
- The architecture contract in `ARCHITECTURE.md` is the integration boundary.
- Advanced replay protection and mitigation are explicitly deferred until the basic pipeline is stable.

## Next action

Create the three feature branches and begin the assigned component implementations against `ARCHITECTURE.md`.

## Do not

- Do not work directly on `main` for components.
- Do not download a large dataset without recording the decision.
- Do not claim model performance until a recorded run produces it.

---

## 2026-10-02: Dataset and Developer B planning update

Developer: Codex assisting Developer B

Branch: `main` (working tree has new untracked planning guides)

### Current state

- User added `data/cic-collection.parquet`, approximately 979 MiB.
- Local commit `80018dc` (`Added dataset`) tracks this raw data, contrary to `data/README.md` and `AGENTS.md` policy.
- User reports canceling the pending push. The current local tracking view still shows `main` one commit ahead of `origin/main`; fetch before concluding whether the commit is unpublished/shared.
- Do not push `80018dc`. Preserve the local dataset and remove it from Git history before publishing, unless a fetch shows the commit is already shared; in that case coordinate rather than rewriting shared history.
- Dataset source, schema, label column, class counts, and relation to CIC-IDS2017 remain unverified. No data inspection, training, or experiment was performed.
- No implementation has started. The developer guides are documentation only.

### Files updated in this planning session

- `docs/REVIEW2_PROJECT_FLOW.md` — reflects the local Parquet candidate and current repository state.
- `docs/DEVELOPER_B_GUIDE.md` — adds Parquet setup/inspection and safe Git cleanup steps.
- `PROJECT_STATUS.md` — updates dataset/Git state and next actions.
- `HANDOFF.md` — records this handoff.
- `.gitignore` — ignores Parquet files directly under `data/` to prevent accidental re-addition after untracking.

### Verification and results

- Read-only checks: `git status --short --branch`, `git log --oneline`, `git ls-files data/cic-collection.parquet`, and `ls -lh data/cic-collection.parquet`.
- Current snapshot: `main` is one commit ahead of `origin/main`; the Parquet file is tracked and remains present locally.
- No test suite or pipeline was run. The Parquet file was not opened or analyzed.

### Next action for the next work session

1. Run `git fetch origin` and confirm whether `80018dc` is absent from shared refs. If unpublished and no one has based work on it, run `git rm --cached -- data/cic-collection.parquet`, add `.gitignore`, and `git commit --amend --no-edit`. Verify the file remains on disk and is no longer tracked. If already shared, coordinate cleanup without rewriting shared history.
2. Switch/create `feature/data-continual` per `TEAM_WORKFLOW.md` after confirming remote branches.
3. Inspect Parquet metadata/schema and dataset provenance without assuming it is CIC-IDS2017; write the data card and select a reproducible subset.
4. Confirm preprocessing, class-ID, and model output-head/replay interfaces before implementing data and continual-learning modules.

---

## 2026-10-02: Developer B dataset audit and schema-loader implementation

Developer: Developer B / Codex

Branch: `feature/data-continual`

### Completed

- Fetched `origin` and confirmed the raw-data commit `80018dc` was absent from remote refs. Removed the Parquet file from the local commit while preserving the local copy, then added `data/*.parquet` to `.gitignore`.
- Created a Python 3.12 virtual environment and installed the direct data dependencies recorded in `requirements.txt`: NumPy 2.5.3, pandas 3.0.6, and PyArrow 25.0.1.
- Audited the complete local Parquet file in read-only batches. The detailed data card is `data/metadata/combined_flow_collection_audit.md`.
- Corrected persistent documentation to describe the input as a combined flow collection with unavailable per-row source provenance, rather than CIC-IDS2017.
- Added `src/data/` schema validation and bounded Parquet batch loading. The default interface validates 57 raw numeric features, removes the three documented unreliable features, then yields 54 numeric features in fixed file order plus `Label` and `ClassLabel`.
- Added synthetic smoke tests in `tests/test_data_loader.py`.

### Observed verification

- `python -m unittest discover -s tests -p 'test_*.py' -v`: 3 passed.
- Strict real-data smoke check: 9,167,581 rows; 57 raw numeric features; 54 retained features; two 128-row batches yielded successfully.
- No processed data, splits, model artifacts, or experiment metrics were created.

### Next action

1. Add deterministic, class-balanced smoke/development sampling and seeded train/validation/test split interfaces with synthetic tests.
2. Add train-only preprocessing that maps documented negative/sentinel values to missing, fits imputation/scaling on training data, and produces `float32` features with fixed integer class IDs.
3. Coordinate Task 1/Task 2 output-head and replay interfaces with Developer A before implementing the continual-learning runner.

### Follow-up: sampling and split implementation

- Added `src/data/sampling.py`: a seeded class-balanced reservoir sampler that scans only `ClassLabel` and retains original Parquet row indices in memory.
- Added `src/data/splitting.py`: a seeded, stratified 70/15/15 splitter that returns non-overlapping positions into a selected sample.
- Added four synthetic tests for sample balance, deterministic seed behavior, insufficient-class rejection, and split completeness/disjointness. Total data-test result: 6 passed.
- Ran a real-file smoke configuration: eight broad classes, 64 rows per class, sampling seed 42, split seed 42. It selected 512 rows and produced 45 train, 10 validation, and 9 test rows per class. No feature values or selected rows were written to disk.

### Follow-up: preprocessing implementation

- Added `src/data/preprocessing.py`. It materializes requested original-file rows in bounded batches; maps negative/non-finite retained values to missing in memory; fits median imputation, mean, and standard deviation on training rows only; transforms all partitions to finite `float32`; and encodes broad labels in a fixed documented order.
- Added a synthetic test that verifies requested row-order preservation, retained-feature selection, finite `float32` output, train centering, and stable class IDs. Total data-test result: 7 passed.
- Ran the real-file 64-per-class smoke sample through preprocessing. Outputs: train `(360, 54)`, validation `(80, 54)`, test `(72, 54)`; every value finite; every split class-balanced; train maximum absolute per-feature mean `5.5e-7` after float32 conversion.
- No raw/processed data, cache, checkpoint, model, or experiment result was written.

---

## 2026-10-02: Developer A + B model-data smoke integration

Developer: Developer A / Model-NIDS

Branch: `feature/model-nids`

### Completed

- Merged the already-merged `feature/data-continual` work into the Developer A branch without modifying `src/data/`.
- Added `tests/test_model_data_integration.py`, an opt-in real-data smoke test.
- The test uses `prepare_sampled_dataset()` and passes `dataset.train.features`, `dataset.train.labels`, `dataset.validation.features`, and `dataset.validation.labels` directly to `TabularTransformerClassifier`.
- It derives `num_features` and `num_classes` from the returned dataset metadata, asserts 54 retained features and eight broad classes, trains for one epoch only, and predicts validation rows.

### Verification

- `python3 -m compileall -q src/data src/models tests` passed.
- `python3 -m unittest discover -s tests -p 'test_*.py' -v` passed: seven data tests; real-data integration test skipped unless explicitly enabled.
- `RUN_REAL_DATA_SMOKE=1 SMOKE_DEVICE=cpu python3 -m unittest discover -s tests -p 'test_model_data_integration.py' -v` skipped cleanly because PyTorch is not installed.

### Remaining blocker

- Add a tested, pinned PyTorch build in the shared environment, then run the opt-in test against `data/cic-collection.parquet`. No model metric or training result exists yet.

---

## 2026-10-02: Unified ownership and confidence novelty baseline

Developer: Unified A+B+C team / Codex

Branch: `feature/unified-novelty`, based on verified
`93492d6dc3145d0d0bcbfd481b6305cda32cd502`

### Completed

- Fetched `origin` and verified that `origin/feature/model-nids` had not advanced
  beyond `93492d6`; `main` and `feature/novelty-poisoning` still lacked the
  model/data implementation.
- Preserved the unrelated untracked `data/data_vis.ipynb` and created the
  focused branch without changing or merging `main`.
- Added the NumPy-only `src/novelty/` package. `detect_unknown()` implements the
  documented strict rule `max_probability < threshold`.
- Added `summarize_predictions()` and `NoveltyResult`. The helper requires
  explicit `class_ids`, maps probability columns to external IDs, and returns
  per-row confidence and unknown decisions.
- Added validation for matrix shape, class-column presence, real finite values,
  `[0, 1]` bounds, approximately unit row sums, scalar finite thresholds in
  `[0, 1]`, and helper class-ID consistency. Empty batches are supported and
  inputs are not changed.
- Added nine synthetic `unittest` cases that do not require PyTorch or data.
- Updated ownership and current-state documentation. A+B+C are one owner of all
  component and shared integration areas.
- Added `docs/IMPLEMENTATION_WALKTHROUGH.md` with the decision record, concrete
  example, verification evidence, limitations, and Review-2 explanation.

### Verification

- Focused novelty suite: 9 passed, 0 failed, 0 skipped.
- Repository `unittest` discovery: 16 passed, 0 failed, 1 skipped. The opt-in
  real-data model/data smoke test remained skipped.
- Novelty compile check: passed.
- `git diff --check`: passed on the complete task diff.
- Optional `pytest` check: could not start because `pytest` is not installed.
- PyTorch is not installed, so no model training was attempted.
- No dataset was loaded and no experiment metric was generated.

### Files changed

- `src/novelty/__init__.py`
- `src/novelty/confidence.py`
- `tests/test_novelty.py`
- `AGENTS.md`
- `TEAM_WORKFLOW.md`
- `ARCHITECTURE.md`
- `PROJECT_CONTEXT.md`
- `PROJECT_STATUS.md`
- `HANDOFF.md`
- `REVIEW2_CHECKLIST.md`
- `README.md`
- `docs/IMPLEMENTATION_WALKTHROUGH.md`
- `docs/DEVELOPER_B_GUIDE.md`
- `docs/REVIEW2_PROJECT_FLOW.md`
- `docs/TEAM_WORKFLOW.md`
- `docs/preprocessing.md`
- `scripts/README.md`
- `src/README.md`
- `src/models/README.md`
- `src/data/__init__.py`
- `requirements.txt`

### Limitations and blockers

- This is a confidence baseline, not evidence of unknown-attack detection.
  Performance requires a held-out unknown-class protocol and recorded metrics.
- No confidence threshold has been calibrated.
- Real-data Transformer training remains unverified because PyTorch is absent.
- Continual tasks, replay, poisoning, mitigation, evaluation, and integration
  remain incomplete.

### Next action

Implement deterministic training-label flipping with a configurable rate and
seed. Preserve original labels and report changed indices and achieved rate.
Do not poison validation/test labels. Stop there before beginning the later
continual-learning, replay, mitigation, and evaluation tasks.

---

## 2026-10-02 23:29 IST: Publish unified feature branch

Developer: Unified A+B+C team / Codex

Branch: `feature/unified-novelty`

### Completed and observed

- Re-read the current project instructions and fetched `origin` before pushing.
- Confirmed local `HEAD` was `8070927`, based on model/data commit `93492d6`.
  No remote `feature/unified-novelty` branch existed before the push.
- Reviewed the push range against `origin/main`. It contained source code,
  tests, and documentation, with no dataset, archive, checkpoint, secret, or
  generated result file.
- Ran the focused novelty tests: 9 passed. Repository `unittest` discovery:
  16 passed and 1 opt-in real-data model test skipped. Compilation and
  `git diff --check HEAD^ HEAD` passed.
- Published with `git push -u origin feature/unified-novelty`. The first remote
  verification returned
  `8070927eddac8bef6e940cd7d2f8c5eda3e9dee6`, matching local `HEAD`.
- Left the unrelated untracked `data/data_vis.ipynb` untouched. No merge to
  `main` or history rewrite was performed.

### Files changed for this handoff

- `PROJECT_STATUS.md`
- `HANDOFF.md`

### Remaining work

Real-data Transformer training is still unverified; the current system lacks
PyTorch. Novelty performance needs held-out unknown-class evaluation. The next
implementation task remains deterministic, configurable training-label
flipping, followed by continual tasks, replay, mitigation, evaluation, and
integration under the unified team's ownership.

---

## 2026-10-02 23:36 IST: Prepare Kaggle GPU smoke notebook

Developer: Unified A+B+C team / Codex

Branch: `feature/unified-novelty`

### Completed

- Fetched `origin` and verified the published unified branch, local `HEAD`,
  and remote branch all pointed to
  `22e797660307bfed62b739d61b5e4c05bcb0a187` before editing. `main`
  remained at `53f7a84` and lacks this combined foundation.
- Added `notebooks/kaggle_review2_smoke.ipynb`, pinned to that full source SHA.
  It explains Kaggle setup, checks the environment without replacing CUDA
  PyTorch, performs a synthetic functional model check, selects the attached
  Parquet path, and invokes existing data/model/novelty APIs.
- The Kaggle path is configured for 64 rows per broad class, seeds 42, a
  stratified split, training-only preprocessing, and one epoch of the tiny
  Transformer on GPU by default. An explicit CPU fallback is available.
- The notebook checks predictions/probabilities and external class-ID mapping,
  records observed classification metrics and novelty flag counts, saves
  indices/preprocessor/checkpoint/history/manifest, and verifies reload
  predictions. All outputs are created at execution time only.
- Updated the implementation walkthrough, README, and project status.

### Verification and current limits

- Local notebook JSON, code syntax, Markdown order, and empty-output validation
  passed: 19 cells, including 9 code cells.
- Novelty tests: 9 passed. Repository `unittest` discovery: 16 passed,
  1 opt-in real-data model test skipped.
- PyTorch and `nbformat` are absent locally; Kaggle GPU execution and the
  attached dataset path are not verified yet. No model metrics or checkpoint
  were generated in this session.
- Unrelated untracked `data/data_vis.ipynb` was preserved.

### Changed files

- `notebooks/kaggle_review2_smoke.ipynb`
- `docs/IMPLEMENTATION_WALKTHROUGH.md`
- `README.md`
- `PROJECT_STATUS.md`
- `HANDOFF.md`

### Next action

Run the notebook on Kaggle with Internet, GPU, and the private attached
`cic-collection.parquet`. Send back the final printed run summary and
`manifest.json`, `metrics.json`, and `training_history.json` for review. Do
not treat confidence flags as unknown-class performance. Choose the next
implementation task after inspecting this execution.

---

## 2026-10-03 00:03 IST: Review smoke and prepare Kaggle clean baseline

Developer: Unified A+B+C team / Codex

Branch: `feature/unified-novelty`

### Completed

- Fetched `origin` and confirmed local and remote unified branch at
  `2b2f5ae6a0e4cbf6a93aa4fbb5318bfddd98160a` before editing. `main`
  remained at `53f7a84`.
- Read the user-supplied, untracked smoke manifest, metrics, history, and
  executed notebook. The JSON values agree, and the executed code cells have
  no error outputs. The files were preserved locally and excluded from Git.
- Recorded the observed Kaggle smoke: source SHA `22e7976`, Tesla T4, Python
  3.12.13, PyTorch 2.10.0+cu128, 360/80/72 rows with 54 features, one-epoch
  train loss `2.1014001899295383`, validation loss `1.9477939367294312`,
  validation accuracy `0.275`, test accuracy `0.25`, macro-F1
  `0.1915376915376915`, and identical checkpoint reload predictions.
  Threshold `0.5` flagged all 72 known-class test rows; this is not
  unknown-attack performance.
- Added a backward-compatible optional `verbose=True` to the existing model
  `fit()` API, printing observed train/validation values per epoch and the
  restored best epoch. The default remains quiet. Documented and tested the
  progress contract; published the source commit
  `68135601a3ddd95832c57bb602f210f1a0e18a43` with a normal push.
- Created `notebooks/kaggle_review2_clean_baseline.ipynb`, pinned to that
  published source commit. It trains a fresh balanced-subset model with the
  requested 2,000/class and 64/4/2/128, batch 128, learning rate 0.001,
  maximum 20 epochs, patience 5, CUDA, and mixed precision disabled.
- The notebook plots train/validation curves, delays test inspection until
  model selection completes, saves metrics and labeled confusion matrix, and
  writes the config, environment, checksum, indices, preprocessing state,
  history, checkpoint, and reload result to a timestamped Kaggle folder.
- Updated the current status, architecture/context, README, and walkthrough.

### Verification

- Model and model-test files compiled locally.
- Repository `unittest` discovery: 16 passed, 1 opt-in real-data model test
  skipped. PyTorch and pytest are absent locally, so the PyTorch progress test
  did not execute here.
- Clean notebook JSON and code syntax: 21 cells, 10 code cells, no stored
  outputs; passed. The larger Kaggle training has not run.

### Changed files

- `src/models/tabular_transformer.py`
- `tests/models/test_tabular_transformer.py`
- `src/models/README.md`
- `ARCHITECTURE.md`
- `notebooks/kaggle_review2_clean_baseline.ipynb`
- `PROJECT_CONTEXT.md`
- `PROJECT_STATUS.md`
- `HANDOFF.md`
- `README.md`
- `docs/IMPLEMENTATION_WALKTHROUGH.md`

### Next action

The clean baseline was subsequently run and reviewed; see the latest entry
below. Do not rerun it merely to invent or replace the observed result.

---

## 2026-10-03: static label-flip comparison prepared

Owner: unified A+B+C. Branch: `feature/unified-novelty`. Before edits,
origin/local HEAD matched at `5e07e3ec681e000a469d23ecbf1e658697e7a815`.
The source API commit `316b817b40f4008af128489ac958d3b232b56049`
was normally published; the notebook checks out that exact SHA.

Reviewed the attached clean-baseline manifest, metrics, 20-epoch history,
executed notebook, curves, and confusion plot. Observed Tesla T4 run from
source `68135601a3ddd95832c57bb602f210f1a0e18a43`: 11,200/2,400/2,400
rows, best epoch 17/20, accuracy `0.8833333333333333`, macro-F1
`0.878016638838182`, training `13.00852884599999` seconds, identical reload.
Benign recall `0.44`: 168/300 predicted attack, 138 as Infiltration. This
balanced-subset result is not production/full-dataset performance. Attached
files under `data/` are untracked, preserved, and excluded from commits.

Implemented `src/poisoning.apply_label_flip` (random and targeted),
`src/evaluation.classification_metrics`, focused synthetic tests, and
`notebooks/kaggle_review2_label_flip.ipynb`. The notebook's 0% control and
5/10/20% attack conditions are **prepared, not run**. It holds the clean
split/preprocessing, architecture, and initial weights fixed; poisons training
labels only; selects by clean validation loss; and saves per-condition audit
and metrics. Targeted rate uses eligible source rows, unlike random's all-row
rate. See `docs/POISONING_EXPERIMENTS.md` for assumptions and limits.

Local focused poisoning/evaluation tests: 6 passed. Full `unittest` discovery:
22 passed and 1 opt-in real-data model test skipped. Notebook JSON, code
syntax, and Markdown-stage checks passed: 15 cells (7 code), empty outputs.
`compileall` and `git diff --check` passed. Local PyTorch/GPU and private
dataset checks remain unavailable; do
not describe the Kaggle poisoning experiment as executed.

Next: run the notebook in Kaggle with Internet, GPU, and the same private
Parquet attachment; return the timestamped folder's manifest, comparison,
condition attack/history/metrics JSON, changed-index files, and executed
notebook. Inspect those outputs before starting clean continual tasks and
balanced replay. Timing backdoors, feature triggers, and replay poisoning are
planned only; mitigation is still unified-team responsibility.

---

## 2026-10-03: clean two-task continual/replay comparison prepared

Owner: unified A+B+C. Branch: `feature/unified-novelty`. Fetched origin;
local/remote branch matched at `a42f464ad1081e7bab21ff11db77992df0a347b9`
before editing. Source API commits
`ca445b425ef96ad4b0d8e214604dec8e49fb0916` and
`b2a0afc697272a9d093780057f38c63ee19340b0` were normally pushed;
the new notebook pins the latter exact SHA. No main merge.

Reviewed untracked `data/poison/manifest.json`, `comparison.json`, and
executed `poison.ipynb` (no error outputs). Seven condition checkpoint reloads
passed, and all initial-weight hashes matched. Clean control accuracy
`0.8833333333333333`, macro-F1 `0.878016638838182`, Benign FPR `0.56`.
Random 20%: 2,240/11,200 changed, accuracy `0.87`, macro-F1
`0.8590190796024948`, Benign FPR `0.67`. Targeted 20%: 1,960/9,800
eligible source rows changed (17.5% of all train rows), accuracy
`0.8533333333333334`, macro-F1 `0.8514854740614037`, attack-to-Benign
rate `0.06571428571428571` versus clean `0.025238095238095237`. The modes
have different denominators; random 10% accuracy and targeted 10% macro-F1
each slightly exceeded control, so do not claim uniform degradation. This is
one seed and one balanced subset, not production performance. All supplied
artifacts remain local and uncommitted.

Implemented `src/continual_learning/tasks.py` and `replay.py`, plus
`src/evaluation/continual.py`. Task builder preserves global IDs 0–7 and fits
the existing preprocessor only on Task 1 training rows. Replay chooses 100
unique Task 1 training examples per old class, seed 42, with original row IDs
and validation/test exclusion checks. Forgetting compares the same old-class
test rows and five old-class F1 scores, retaining negative improvements.
Synthetic tests cover boundaries, row disjointness, future-data exclusion,
replay determinism, balance, copies, and forgetting.
Focused four-test continual suite passed. Full `unittest` discovery: 26
passed, 1 opt-in real-data model test skipped; `compileall`, notebook
JSON/code syntax/Markdown/empty-output checks (23 cells, 11 code), and
`git diff --check` passed.

Created `notebooks/kaggle_review2_continual_replay.ipynb` with a fresh
five-output Task 1 model, independent Task 2 checkpoint reloads, identical
expanded-initial-state hash assertion, sequential/replay arms, clean seen-class
validation, test-after-selection evaluation, training-work counts, curves,
row/preprocessing records, checkpoints, and reload verification. This is
**prepared, not executed**; no continual/forgetting outcome is known. The
local environment has no PyTorch/CUDA or private attached Kaggle input.

Next: run the notebook on Kaggle with Internet, GPU, and the same private
Parquet input. Return the executed notebook and timestamped folder: manifest,
summary, Task 1/arm histories and metrics, row/replay indices and copied
replay buffer, preprocessing
state, training curves, and checkpoint reload flags. Review observed results
before replay poisoning, mitigation, timing backdoors, or full-data streaming.

---

## 2026-10-03: replay-label poisoning and consistency gate prepared

Owner: unified A+B+C. Branch: `feature/unified-novelty`. Fetched origin;
local/remote HEAD matched `a4676725d06aa993764b91f7f342d5a5b1f330ad`
before edits. Published the source API commit
`c636331fc9cdf2361357fd901d9c93411e3f59ba` normally. The new Kaggle
notebook pins that full SHA. No main merge.

Inspected untracked `data/continual/continual.ipynb`, `manifest.json`,
`summary.json`, `replay_buffer.npz`, and `training_curves.png`. The executed
notebook has no error outputs; the buffer has 500 unique rows, 100 per old
class. Source `b2a0afc697272a9d093780057f38c63ee19340b0`, Tesla T4,
Python 3.13.15, PyTorch 2.11.0+cu128, NumPy 2.1.3, PyArrow 23.0.1. Prior
static-poisoning runtime used Python 3.12.13, PyTorch 2.10.0+cu128, NumPy
2.0.2, PyArrow 24.0.0. No Task 1 checkpoint was attached, so the new
notebook trains it afresh and compares conditions only within its own run.

Observed old-class accuracy before Task 2 `0.9686666666666667`. Sequential
after: old `0`, new `0.9377777777777778`, combined
`0.3516666666666667`, Benign FPR `1.0`. Clean replay after: old
`0.6826666666666666`, new `0.9433333333333334`, combined
`0.7804166666666666`, Benign FPR `0.9366666666666666`. Sequential/replay
used 264/444 optimizer steps. Replay helped relative to sequential but
Benign forgetting remains severe, and this single-seed balanced-subset
baseline is not deployment-ready or equal-compute evidence.

New `src/mitigation/label_consistency.py` scores supplied replay labels
through explicit teacher `class_ids`, calibrates a fixed 95th-percentile
threshold on clean Task 1 validation, and quarantines scores strictly above
it. It returns copies and `None` if no row remains. The gate has no clean
simulator labels or changed-mask input. New
`src/evaluation/replay_gate.py` receives those hidden records *afterward*
to audit poison rejection, clean false rejection, retained poison fraction,
and retained counts by supplied class. Synthetic tests cover mapping,
calibration, non-mutation, equality at threshold, empty replay, invalid
inputs, and random/targeted budget accounting.

`notebooks/kaggle_review2_replay_poison_mitigation.ipynb` prepares six paired
conditions: clean ± gate; random 20% ± gate; targeted 20% old attacks→Benign
± gate. Random budget is 100/500; targeted is 80/400 eligible, or 16% of
the full buffer. Each pair reuses the identical corrupted labels. All six
Task 2 models independently reload one new clean Task 1 checkpoint, verify
matching expanded initial weights, and use the same clean Task 2 data and
trusted validation. Condition outputs include audit arrays/JSON, history,
metrics, configuration, checkpoint/reload, timing, and optimizer steps;
shared output includes the calibration, split/preprocessing/replay records,
comparison, and plot. This notebook is **prepared, not executed**.

Next: run the notebook on Kaggle with Internet, GPU, and the same private
Parquet file. Return the executed notebook and timestamped output folder,
especially `manifest.json`, `comparison.json`, `calibration.json`,
Task 1 metrics/history, each condition's `attack.json`, `gate_audit.json`,
`replay_audit.npz`, `history.json`, `metrics.json`, and checkpoint reload
flags, plus curves. Review observed results before stronger replay balancing,
timing backdoors, novelty evaluation, or full-data streaming. Do not infer
that filtering improves performance without the run.

Verification: focused label-consistency suite 5 passed; repository
`unittest` discovery 31 passed and 1 opt-in real-data check skipped.
`compileall`, notebook JSON/code syntax/Markdown-stage/empty-output checks
(21 cells, 10 code), and `git diff --check` passed. Local PyTorch/CUDA and
the private Kaggle input were unavailable, so no local GPU result is claimed.

---

## 2026-10-03: clean replay exposure and held-out novelty prepared

Owner: unified A+B+C. Branch: `feature/unified-novelty`. Fetched origin and
verified local/remote HEAD `3b68e0347f4e9229a9a12143c4fc77d2519fcbb0`
before editing. Source API commit
`4e46f7f8ded77612210237e8dc9f6ea8e83730da` was normally pushed and is
pinned by the new notebook. No main merge; local `data/` attachments preserved.

Reviewed the untracked mitigation manifest, comparison, calibration,
executed notebook, and plot. The six-condition run used source
`c636331fc9cdf2361357fd901d9c93411e3f59ba` on Tesla T4 with Python
3.13.15 and PyTorch 2.11.0+cu128. All six expansion weight hashes matched,
and six checkpoint reloads passed. Combined accuracy clean: .7804166667
unfiltered, .81375 filtered; random20: .81625/.7858333333; targeted20:
.81625/.8279166667. The gate rejected 100/100 random and 80/80 targeted
poisoned rows, but falsely rejected about 4–5% of clean candidates. Benign
FPR remained .90–.9533. Optimizer steps differed across paired conditions:
clean 444/444, random20 370/648, targeted20 444/504. This single-seed
result does not establish consistent downstream benefit or deployment safety.

Implemented optional class-balanced sampling in `src.models`, backed by
NumPy-only inverse-frequency weights in `src.training`. Default shuffled
training is unchanged. The balanced sampler draws with replacement for the
combined Task 2 + replay row count; it does not increase the 500 unique stored
replay examples. History records seed, expected and actual class exposure,
unique replay rows drawn, total training draws, optimizer steps, and advancing
generator-state hashes. Added fifth-percentile known-validation confidence
calibration in `src.novelty` and held-out binary/per-class novelty metrics in
`src.evaluation`.

Created `notebooks/kaggle_review2_replay_balance_novelty.ipynb`, pinning the
published source. It rebuilds the Task-1-training-only preprocessor, trains
one five-output Task 1 model, evaluates known Task 1 and unseen Task 2 test
rows using a threshold fixed from Task 1 validation **before expansion**,
then compares shuffled/balanced Task 2 arms from equal expanded weights and
unchanged replay memory. Each arm's checkpoint uses clean validation loss;
the preferred strategy is chosen by predeclared seen-class validation
macro-F1 and Benign FPR tie-breaker, never by test/novelty results. It saves
source/environment/checksum, row/replay metadata, histories, exposure,
metrics, checkpoints, plots, and reload checks. Notebook is **not executed**;
there are no balanced-exposure or held-out-novelty results yet.

Local full `unittest` discovery: 39 tests, 36 passed, three skipped (two
PyTorch-dependent sampler checks, one opt-in real-data integration check).
Source compilation passed. Notebook JSON/Python code syntax and empty-output
checks passed; local PyTorch/CUDA and private dataset are unavailable. Kaggle
will execute the torch-dependent synthetic tests before dataset scanning.

Next: run the notebook in Kaggle with Internet, T4 GPU, and the same private
`cic-collection.parquet`. Return executed notebook and timestamped folder:
`manifest.json`, `strategy_selection.json`, `novelty.json`, `test_summary.json`,
both arms' `history.json`, `validation_metrics.json`, `test_metrics.json`,
checkpoint reload flags, `row_indices.npz`, `replay_buffer.npz`, and plot.
Review actual results, then prepare streaming/throughput benchmarking. Do not
claim novelty performance or balanced-sampling improvement before the run.

---

## 2026-10-03: Kaggle test discovery corrected

Owner: unified A+B+C; branch `feature/unified-novelty`. Fetched origin and
verified local/remote HEAD
`0315bc041acec0a69629fda0764d43836dda576c` before editing. Preserved
all untracked `data/` artifacts and existing notebooks.

User-reported Kaggle failure: the first attempted execution of
`kaggle_review2_replay_balance_novelty.ipynb` stopped at its synthetic-test
cell with `ModuleNotFoundError` for dotted `tests.test_training_sampling`
and `tests.test_novelty_evaluation`. The files are present at the pinned
source revision, but `tests/` has no `__init__.py`; the import form is not
portable to that Kaggle environment. This is a notebook bootstrap failure,
not evidence about balanced replay or unknown-attack detection.

The notebook now runs two separate exact-pattern `unittest discover`
subprocesses from `REPO_DIR`, with `check=True`, captured/printed output,
and an explicit nonzero discovered-test count for each. It still pins source
`4e46f7f8ded77612210237e8dc9f6ea8e83730da`; data, model, sampler,
threshold, seeds, and strategy-selection settings are unchanged. No source
API edit was needed.

Local verification: sampler pattern discovered 4 tests, 2 passed and 2
PyTorch-dependent skipped; novelty pattern discovered 3 tests, all passed.
Full `unittest` discovery ran 39 tests, 36 passed and 3 skipped. Notebook
JSON, Python syntax in all 11 code cells, preceding Markdown, empty outputs,
and unchanged pinned SHA passed; `git diff --check` passed. Local PyTorch,
CUDA, private Parquet data, and a corrected Kaggle rerun were unavailable.

Next: import the updated notebook into a **fresh** Kaggle Internet/GPU
session, attach the reviewed Parquet file, and run top to bottom. Confirm both
synthetic suites report nonzero tests and pass before interpreting Task 1,
balanced-replay, or held-out novelty outputs. Return executed notebook and
timestamped output folder. Only after that review proceed to streaming
preparation/throughput benchmarking. Do not claim this failed run produced
experiment metrics.

---

## 2026-10-03: full-row preparation and throughput notebook prepared

Owner: unified A+B+C; branch `feature/unified-novelty`. Started by fetching
origin and verifying local/remote HEAD
`8c09efccb5593247a8c57f34e200f262066e9400`. All untracked `data/`
attachments, previous notebooks, model, and small-array APIs were preserved.

Reviewed untracked executed `data/replay/replay.ipynb`, `manifest.json`,
`test_summary.json`, `strategy_selection.json`, `novelty.json`, and plot.
There were no notebook errors; 4 sampler and 3 novelty Kaggle tests passed;
Task 1 and both Task 2 checkpoint reload predictions matched. Validation
macro-F1 selected balanced replay (.8529214 vs .7541214 shuffled). Test
combined accuracy balanced/shuffled .8508333/.7804167; old
.8353333/.6826667; new .8766667/.9433333. Balanced Benign FPR remained
.6166667. Balanced/shuffled took 370/444 optimizer steps, so this is a
single-seed, unequal-work, bounded-subset comparison. Fifth-percentile
known-validation threshold .9247980 flagged only 170/900 unseen Task 2
rows (unknown recall .1888889) and 68/1500 known rows (false rejection
.0453333). Confidence novelty is weak here, not established broadly.

New `src/data/large_data.py` prepares deterministic full-row 70/15/15
class-stratified partitions with original row IDs and split indices; scans
Parquet features in bounded batches into one row-aligned float32 disk store;
excludes `Label`/`ClassLabel` as features; and fits independent static and
Task-1-only continual states. Approximate medians use a seeded, recorded
200,000-row training-only reservoir; population means/stds stream all
authorized training rows. Completed stages are restartable. New
`src/training/disk_backed.py` trains from bounded transformed batches with
the existing Transformer, class IDs, checkpoint format, and validation-loss
early stopping. Deterministic class-bucket balanced draws require no
full-data float64 weight vector. No large-data model training was run.

Published source commits `6badb33ec3b5c074a174bf3b188571daf53e8f5f`
and `5c0d519596bc6c4b99948126443635c2f7eb15f1`; new
`notebooks/kaggle_review2_large_data_prepare.ipynb` pins the latter. It
verifies GPU, source SHA, input checksum, row count and output space, saves
complete partitions and both fit states, then measures one epoch on a fixed
training-only sample of up to 100,000 naturally distributed rows at batches
128 and 256. It records measured throughput, RAM/GPU peaks, optimizer steps,
storage, and clearly labeled 1.5x-headroom full-epoch estimates, plus a
provisional configuration. No test metrics are opened. Persist full output
folder as a Kaggle version/private dataset, never in Git.

Verification: `python3 -m unittest discover -s tests -p 'test_large_data.py'
-v` ran 4 synthetic tests, 3 passed and 1 PyTorch-dependent skipped.
Repository `unittest` discovery ran 43 tests, 39 passed and 4 skipped.
Source compilation, notebook JSON/Python syntax/empty-output checks, and
`git diff --check` passed. Local PyTorch/CUDA were unavailable. The full
9,167,581-row preparation and T4 throughput benchmark are **not executed**.

Next: run the new notebook on Kaggle with Internet, T4, and the exact
reviewed input. Return executed notebook plus `manifest.json`,
`split_manifest.json`, `features_manifest.json`,
`preprocessing_static.json`, `preprocessing_continual.json`,
`benchmark_results.json`, `recommended_full_run.json`, and file-size report.
Retain all `.npy` files privately for the next session. Review disk/throughput
measurements before real full-data training; do not start the full suite in
this preparation phase.

---

## 2026-10-03: full clean static large-data run prepared

Owner: unified A+B+C; branch `feature/unified-novelty`. Fetched origin and
verified local/remote HEAD
`96b8485b3d7f36fcb5d910fce6e5d4149de30a87` before edits. Preserved
all untracked `data/` attachments, existing notebooks, prepared data, and
small-array APIs. No main merge or generated artifact commit.

Inspected untracked `data/preparation/preparation.ipynb` and every supplied
preparation JSON. Eight code cells executed with no error outputs. Source
`5c0d519596bc6c4b99948126443635c2f7eb15f1`, Tesla T4/Python
3.13.15/PyTorch 2.11.0+cu128, reviewed input checksum. Observed split:
6,417,308 train, 1,375,139 validation, 1,375,134 test (all 9,167,581).
Static state fitted 6,417,308 training rows; continual state fitted
6,347,232 Task-1 training rows; each used a seed-42 200,000-row median
reservoir. The prepared output totaled about 2.146 GB. The benchmark
measured 22,196.07 rows/s at batch256 on 100,000 naturally distributed
training rows (391 steps, 4.5053 s). `433.68` seconds is a 1.5x-headroom
**estimate** for a full training epoch and **excludes validation**. Large
`.npy` arrays were not attached locally, so only the supplied run records
were inspected; no full-data model result exists.

Added `src/training/full_run.py` with atomic per-epoch latest/best
checkpoints. Latest includes current model, optimizer, scaler, Python/NumPy/
Torch CPU/CUDA RNG, completed epoch, history, best weights/loss, and patience;
resume verifies config, class IDs, prepared-data identity, and runtime.
Interrupted epochs replay from the previous boundary, not mid-epoch. It
prints batch progress and measured train/validation/remaining timings.
`src/evaluation/streaming.py` accumulates bounded test confusion counts and
reports accuracy, macro-F1, balanced accuracy, per-class P/R/F1/support,
and Benign FPR. The original small-array model API remains unchanged.
Published API source: `91a26a1c13a563a7628e217f357dc27f73874784`.

New `notebooks/kaggle_review2_full_clean_training.ipynb` pins that source.
It explicitly selects a read-only prepared folder from current working or
attached Kaggle input, checks manifests/shapes/dtypes/split coverage/
preprocessing scope and hashes its raw features, labels, splits and state.
It creates a separate writable run folder and supports resume from a prior
working folder or attached run output. Configuration is a fresh eight-class
hidden64/four-head/two-layer/MLP128 model, batch256, lr.001, seed42,
mixed precision off, ordinary shuffled full training, max20 epochs,
patience5. Full validation loss chooses best; only then does one bounded
complete test inference occur. Artifacts include latest/best, per-epoch
history/manifest, config/environment/data hashes, reload audit, plots,
test metrics and final manifest. **Notebook unexecuted; no full-data
accuracy or test metric is claimed.**

Local focused streaming metrics: 2 passed. Resume/header suite: 1 passed,
1 PyTorch-dependent equivalence test skipped. Full `unittest`: 47 tests,
42 passed, 5 skipped. Source compilation, notebook JSON/code syntax,
Markdown and empty-output checks, and `git diff --check` passed. PyTorch/GPU
and the actual prepared arrays were unavailable locally; the Kaggle notebook
runs focused PyTorch tests before training.

Next: import the full clean notebook into Kaggle with Internet/T4, set
`PREPARED_DIR` to the existing `/kaggle/working/review2_large_prepare_...`
or attached private input folder, and run. Save the run folder after every
completed epoch. If interrupted, attach that run folder and set
`RESUME_FROM`; keep the exact prepared input attached. Return executed
notebook plus `final_manifest.json`, `training_manifest.json`, `history.json`,
`run_config.json`, `run_environment.json`, `preprocessing_continual.json`,
`reload_verification.json`,
`test_metrics.json`, `training_curves.png`, `confusion_matrix.png`, and
checkpoint metadata/files privately. Inspect clean full-data results before
separate continual/replay and poisoning runs.

---

## 2026-10-03: full clean result reviewed; full Task 1 prepared

Owner: unified A+B+C. Started on `feature/unified-novelty` at
`3cd90ac217c0842766b5849e2fc343a9a3d2c71f`. Fetched origin and
preserved untracked `data/` attachments, the separate faculty-presentation
work, and local interfaces. No main merge, dataset/checkpoint commit, or
Task 2 training.

Reviewed the supplied full-clean final/training manifests, history,
test metrics, reload JSON, environment, and original raw-count/training
plots. The separate `data/full-train.ipynb` has three `NameError` outputs,
so it is not proof of the completed v2 run; results below come from its
completed v2 artifacts. Source `91a26a1c13a563a7628e217f357dc27f73874784`;
19 epochs, best validation-loss epoch 14, measured train+validation
`104.2950168455` minutes. Full static test accuracy `0.9828620338090688`,
macro-F1 `0.757755762638612`, balanced accuracy `0.7291289708762002`,
Benign FPR `0.00445669840657261`; Infiltration recall `0.059108799550182736`
and Webattack recall `0.12694877505567928`. The high overall accuracy is
not sufficient: most Infiltration/Webattack rows were called Benign. The
supplied reload artifact reports best-state and prediction agreement, but
no independent local checkpoint load occurred without PyTorch.

Published `src/evaluation/plots.py` and synthetic tests at
`cdc9bc09bf1f98ca1c5f3bc78372637ac655926d`. It reads existing metrics
JSON, preserves the original raw plot, and writes separate raw-count,
annotated row-normalized, and recall/F1 charts. A local test generated
these three views from the attached counts in a temporary folder; it did
not open the dataset or repeat inference. Published deterministic
`select_disk_replay_rows` and tests at
`e6ebfcf7a9beec4d2b8fd5245e7c62314b9fadf9`; it accepts only a
training view and returns 100 unique original positions per old class.

New `notebooks/kaggle_review2_full_task1_training.ipynb` pins the second
source SHA. It validates existing prepared arrays and the Task-1-only
preprocessor, hashes inputs, initializes a fresh five-output model, trains
all 6,347,232 old-class training rows with full old-class validation and
durable epoch-boundary resume, verifies best reload, and saves a clean
500-row replay buffer with transformed features and original IDs. After
validation selection it calibrates a fifth-percentile threshold from Task 1
validation only, then performs bounded old-class test and unseen Task 2 test
inference with the **unexpanded** head. Task 2 rows never train Task 1,
fit preprocessing, or set the threshold. The notebook is unexecuted;
no Task 1 or full-data novelty metric is claimed.

Verification: focused plot tests 3 passed; disk replay tests 2 passed.
Full `python3 -m unittest discover -s tests -p 'test_*.py' -q`: 52 tests,
47 passed, 5 skipped and zero failed. `python3 -m compileall -q src tests`,
17-cell/8-code notebook JSON/code syntax/Markdown-stage/empty-output/pin
checks, and `git diff --check` passed. Local PyTorch
is unavailable, so CPU resume-equivalence and Kaggle GPU/data execution
remain unverified here. Preserve the whole `/kaggle/working/full_task1_...`
folder as a private Kaggle version/input. Return executed notebook,
`final_manifest.json`, `training_manifest.json`, `history.json`,
`run_config.json`, `run_environment.json`, `reload_verification.json`,
`replay_manifest.json`, `task1_test_metrics.json`, `novelty.json`,
`training_curves.png`, and three Task 1 class plots. Keep
`replay_buffer.npz`, `best.pt`, and `latest.pt` privately for later Task 2
and resumption. Next: inspect actual Task 1 outputs before designing the
separate Task 2 continuation; do not infer results from the static model.

---

## 2026-10-03: pre-run Task 1 epoch-cap amendment

The unified branch was clean except for preserved untracked `data/`
attachments. After fetching origin, local/remote HEAD matched
`9a33b588f0319a8042f68d2883490ec1e7da024b`. The full Task 1 notebook
was **not run** and no Task 1 results were observed. Set only its maximum
epochs from 20 to 12 as a predeclared Kaggle time-budget decision. Keep
patience 5, all other hyperparameters, Task-1-only preprocessing, full
old-class validation-loss checkpoint selection, durable per-epoch
`latest.pt`/`best.pt`, strict resume, bounded test/novelty evaluation, and
500-row clean replay export unchanged. A checkpoint made with the former
20-epoch configuration will fail resume compatibility rather than silently
continue under the new cap. Do not alter the completed eight-class static
baseline or use its test metrics to tune Task 1.

Changed files: the Task 1 notebook, `PROJECT_STATUS.md`, this handoff, and
`docs/IMPLEMENTATION_WALKTHROUGH.md`. Notebook JSON/code syntax,
Markdown-before-code, empty outputs, published source pin, max12/patience5,
resume/evaluation/replay references passed for 17 cells and 8 code cells.
`python3 -m unittest discover -s tests -p 'test_*.py' -q` ran 52 tests:
47 passed, 5 skipped (PyTorch-dependent or opt-in real-data checks), zero
failed. `python3 -m compileall -q src tests` and `git diff --check` passed.
No local PyTorch/GPU Task 1 run was attempted. Next: run this revised
notebook on Kaggle with the original prepared folder, preserve the whole
Task 1 output directory, and review the actual results before Task 2.
