# Handoff

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
