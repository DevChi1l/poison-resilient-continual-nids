# Developer B: Data and Continual Learning Step-by-Step Guide

> Historical planning guide. Its separate A/B/C ownership rules, repository
> snapshot, and immediate actions are superseded by `TEAM_WORKFLOW.md`,
> `PROJECT_STATUS.md`, and `HANDOFF.md`. Developers A+B+C now form one owner;
> do not wait for or assign work to a former lettered owner. Retain this file
> only for the rationale and commands that remain technically useful.

This guide turns the checked-in Review-2 architecture into a practical work sequence for Developer B. It covers setup, ownership, dataset preparation, data interfaces, continual tasks, verification, integration, and handoff. It does not claim that any of these implementation steps are already complete.

## 0. Know the repository state and team rules

The checked-out project currently contains planning documents and a local `data/cic-collection.parquet` file (about 979 MiB). No `requirements.txt`, `pyproject.toml`, `src/` implementation, pipeline entry point, tests, or experiment results are present. The current local Git state shows `main` one commit ahead of `origin/main`; commit `80018dc` (`Added dataset`) tracks the Parquet file. The user reports canceling the pending push. This conflicts with `data/README.md` and `AGENTS.md`, which prohibit committing source datasets. **Do not push this commit.** The file is available locally for Developer B after it is safely untracked. `PROJECT_STATUS.md` records Python 3.14.6 on the initialization machine, with NumPy and pandas available there but PyTorch and scikit-learn absent; that is machine-specific, not a portable setup guarantee.

The checked-in workflow assigns Developer B:

- `src/data/`: loading, validation, preprocessing, splits, task data.
- `src/continual_learning/`: Task 1/Task 2 driver and optional replay interface.
- `data/metadata/`: dataset and preprocessing metadata, never raw datasets.
- Shared integration files: `run_pipeline.py`, `README.md`, `configs/`, dependency manifest, `PROJECT_STATUS.md`, `ARCHITECTURE.md`, and `HANDOFF.md`.

Do not edit Developer A's model/training modules or Developer C's novelty/poisoning/evaluation modules. Ask for or document an interface where a dependency is underspecified. The repository docs have three technical developer roles; reconcile this with any four-person team assignment before changing ownership files.

## 1. Start from the right Git state

Run these from the cloned repository root. Do not push the current local commit until the dataset has been removed from the commit while preserving its local copy:

```bash
git status --short --branch
git log --oneline -10
git remote -v
git fetch origin
git branch -a
```

The repository policy is to keep raw datasets out of Git. The current `data/cic-collection.parquet` is already tracked in the latest local commit. First fetch and verify whether that commit has been published or shared. If it remains unpushed and no teammate has based work on it, untrack the file while keeping it on disk, then amend that local commit:

```bash
git fetch origin
git status --short --branch
git rm --cached -- data/cic-collection.parquet
```

Add this rule to `.gitignore` if it is not already present:

```gitignore
data/*.parquet
```

Then update the local unpushed commit and confirm the file remains present but is no longer tracked:

```bash
git add .gitignore
git commit --amend --no-edit
git status --short --branch
git ls-files data
ls -lh data/cic-collection.parquet
```

`git rm --cached` preserves the local file. Do not use `git rm` without `--cached`. If fetching shows that the commit is already on a remote or a teammate has based work on it, do not rewrite shared history; coordinate the cleanup with the team before pushing further changes.

Preserve any local changes before switching branches. The documented B branch is `feature/data-continual`, but the current clone initially showed only `main` and `origin/main`. After fetching, if `origin/feature/data-continual` exists, check it out tracking the remote branch:

```bash
git switch --track origin/feature/data-continual
```

If no such remote branch exists and the team agrees you should create it from updated main:

```bash
git switch main
git pull --ff-only origin main
git switch -c feature/data-continual
```

Do not work directly on `main`. Check `git status --short --branch` again after switching. Coordinate before pushing or changing a branch another teammate is using.

## 2. Establish the Python environment

There is no dependency manifest yet, so do not assume that `pip install -r requirements.txt` works. First agree on one Python version supported by the selected PyTorch and scikit-learn builds and use the same version in local development and Colab where practical. The initialization host's Python 3.14.6 was not checked against either package. A Python 3.12 environment is a reasonable compatibility starting point, but verify current wheels for the target OS/runtime before standardizing it. PyTorch's install selector is the authority for the matching current install command: [PyTorch Start Locally](https://docs.pytorch.org/get-started/locally/).

Linux/macOS example, replacing `python3.12` with the agreed installed interpreter:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell activation is:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Once the team has selected compatible versions, the likely direct package set for the first integrated pipeline is:

- `numpy` and `pandas` for arrays and CSV handling.
- `pyarrow` (or another pandas-supported Parquet engine) to read the supplied `.parquet` file through pandas.
- `scikit-learn` for train/validation/test split utilities and standard preprocessing/metrics if the implementation uses them.
- `torch` for Developer A's model and training code.
- `PyYAML` only if configs are YAML.
- `pytest` for the repository's automated smoke/unit tests.

Install PyTorch with the command from its official selector for the target platform and compute choice. A Colab GPU runtime often already has a compatible PyTorch build; check `torch.cuda.is_available()` before replacing it. Use the selected CPU build for local smoke runs when appropriate. After A and B agree on versions, record direct dependencies in the shared dependency manifest, and record exact versions used for an experiment. Do not invent version pins before resolving Python/runtime compatibility. The isolated-environment approach is also recommended by the [scikit-learn installation guide](https://scikit-learn.org/stable/install.html).

Basic environment inspection:

```bash
python --version
python -m pip --version
python -c "import numpy, pandas; print(numpy.__version__, pandas.__version__)"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The last command works only after PyTorch has been installed. Do not install GPU driver/CUDA system packages inside Colab as a first step; use the runtime's provided accelerator and the matching PyTorch build.

## 3. Freeze the data/model contract before coding

Read `ARCHITECTURE.md` and coordinate with A on:

1. Input feature dtype and shape: `float32`, `(n_samples, n_features)`.
2. Target dtype and shape: integer class IDs, `(n_samples,)`.
3. Stable ordered `feature_names` and `class_names` mapping.
4. How the model's `class_ids` and probability columns behave after new classes arrive.
5. Whether `fit(..., replay=...)` accepts arrays, batches, or a loader.
6. How preprocessing state is saved/reloaded and supplied at inference.

The current architecture document does not define the model output-head expansion mechanism needed for class-incremental learning. Agree on it before connecting Task 2, and update the architecture contract before implementation if it changes. Never silently remap existing class IDs.

## 4. Select and document the first dataset scope

The project blueprint chooses CIC-IDS2017 as the initial primary dataset. A local file named `data/cic-collection.parquet` has been added, but its source, schema, label coverage, and relationship to CIC-IDS2017 have not been verified. Treat it as a candidate input, not as a confirmed CIC-IDS2017 subset. Identify and inspect it before deciding whether it is suitable. Do not download all three candidate datasets or start with the largest available files. Select a manageable, reproducible subset for the first end-to-end run.

Before processing, inspect:

- File names, format (Parquet or CSV), columns/schema, row counts, and file sizes. For Parquet, inspect row-group/schema metadata before loading the full table when possible. For CSV, also verify encoding and delimiter.
- Target column and exact raw label strings, including whitespace/case variants.
- Dtypes, missing values, `NaN`, positive/negative infinity, and duplicate records.
- Identifier, timestamp, source/destination address, and other fields that could leak labels or split membership.
- Class imbalance and whether selected classes have enough rows for the agreed partitions/tasks.

Record in a dataset card under `data/metadata/`:

- Dataset name, source URL/provider, release/version, access date, license/usage notes if known.
- Original file names and cryptographic checksums where available.
- Subset selection rule, random seed, source row counts and selected row counts.
- Label column, raw labels, normalized labels, and class-to-ID mapping.
- Excluded columns and reason for exclusion.
- Invalid-row/duplicate policy, feature transform, split protocol, and known limitations.

Keep raw data outside Git or in an approved ignored data location. Check ignore rules before placing files. Do not commit data, checkpoints, secrets, or bulk result files.

## 5. Implement the smallest reliable data path

Suggested module responsibilities (follow existing naming if the team creates a different agreed layout):

```text
src/data/
  loading.py          read configured CSV/Parquet paths and selected columns
  validation.py       check schema, labels, finite numeric values
  preprocessing.py    fit transform on train only, apply to other splits
  splitting.py        seeded split helpers and split metadata
  types.py            PreparedDataset/Split data structures
src/continual_learning/
  tasks.py            deterministic Task 1/Task 2 definition and extraction
  runner.py           sequence training/evaluation through model interface
```

Avoid building abstractions before they solve a concrete need. Keep file paths/configuration outside the code. Do not load all CIC-IDS2017 files by default: support a configured file list and a deterministic row cap or sampling strategy for smoke/medium/final runs.

Data processing sequence:

1. Read only the configured files and required columns where feasible. Select the reader from the configured/validated file format; for Parquet, ensure a supported engine such as `pyarrow` is installed.
2. Normalize label strings in a documented, deterministic way; reject or explicitly map unknown label values rather than silently dropping them.
3. Select feature columns without label leakage. Drop identifiers and raw labels from `X`.
4. Convert features to numeric values under an explicit policy. Make missing and infinite values finite or drop affected rows according to the recorded policy.
5. Create train/validation/test indices with a fixed seed and appropriate class coverage. If data ordering or repeated capture sessions could leak between partitions, discuss a group/time-aware split; do not blindly random-split near-duplicates.
6. Fit imputers/scalers/encoders on training data only. Apply exactly the same fitted transform to validation and test.
7. Return contiguous `float32` feature matrices and integer label vectors. Preserve feature order and raw readable class names.
8. Record class counts per split and transformation settings in metadata.

For Review 2, scikit-learn's standard split/scaler components are acceptable if they match the agreed protocol. The key research-integrity requirement is preventing fit/test leakage and documenting the split.

## 6. Define Task 1 and Task 2

Start with class-incremental tasks that answer the research question: Task 1 contains a selected subset of known classes; Task 2 introduces held-out classes. The exact class choices depend on actual label counts and must be recorded, not guessed in advance.

Define and save:

- Task class membership, task order, and global class IDs.
- Training/validation/test row indices or reproducible split seed and selection rule.
- Per-class row counts in every partition and task.
- Whether Task 1/Task 2 share the same feature transform (they should in the basic benchmark if fit on the permitted training data).
- Evaluation sets that remain fixed before/after Task 2.

Evaluation order:

1. Train on Task 1 training rows.
2. Evaluate Task 1 classes and the Task 1 held-out test set; save `task1_before`.
3. Evaluate Task 2-only classes before they are learned as a novelty/unknown check when the interface supports it.
4. Update on Task 2 training rows, optionally with a small replay buffer only after simple sequential fine-tuning works.
5. Evaluate the same Task 1 test rows; save `task1_after`.
6. Evaluate Task 2 test rows; save `task2_after`.
7. Report forgetting as the change in Task 1 score, with sign convention stated. Keep macro-F1/per-class outcomes alongside the summary.

Do not train or tune on test rows. A future class used as “unknown” during Task 1 evaluation must not also be included in Task 1 training.

## 7. Add a tiny smoke path and focused tests

Use deterministic synthetic data only to check code wiring. It should exercise schema validation, numeric conversion, label mapping, seeded splitting, train-only fitting, output shapes/dtypes, stable feature/class order, and task membership without reading the real dataset. Label synthetic smoke outputs clearly and never include them in NIDS performance claims.

Add focused tests for at least:

- Missing/extra expected columns and malformed labels.
- NaN/inf and duplicate handling under the chosen policy.
- Label-to-ID consistency and unseen-label behavior.
- Reproducible split/task construction with fixed seeds.
- No train/validation/test index overlap.
- Preprocessor fit only on training rows and consistent transform dimensions.
- Task class disjointness/coverage according to the chosen class-incremental design.
- Canonical arrays' rank, shape, dtype, finiteness, and feature order.

Run the tests when code exists and report exact commands and outcomes. Do not claim the dataset-scale pipeline passed based only on synthetic data.

## 8. Make runs configurable and resumable

Coordinate the shared config with A and C. It should eventually capture:

```yaml
seed: 42
data:
  paths: []                 # actual paths supplied by each user/environment
  label_column: null
  sample_limit: null
  split_seed: 42
  task_definition: null
training:
  batch_size: null
  epochs: null
  device: auto
  num_workers: 0
output_dir: results/
```

These are a schema sketch, not a runnable config. Replace nulls with validated values after code/data/API decisions. Keep raw paths out of shared config if they are machine-specific; support environment variables or a local ignored config. Cache only useful, reproducible intermediate artifacts and store their source checksum, preprocessing version, split seed, and transform metadata. Never trust a cache if those inputs differ.

Support at least three scales:

1. **Smoke:** synthetic/tiny input, CPU, minimum epochs/rows to validate wiring.
2. **Development:** manageable real subset, quick pipeline debugging.
3. **Experiment:** documented real subset on GPU with chosen epoch/batch/model configuration.

The actual config values and results must be captured in the manifest. Do not describe the sketch command/config as runnable until implemented and observed.

## 9. Integrate after individual interfaces work

Developer B is the integration coordinator under current repository rules. Once A and C publish working components:

1. Integrate one stage at a time against `ARCHITECTURE.md`.
2. Keep module ownership intact; request changes from that component's owner instead of patching their core files.
3. Add the one-command `run_pipeline.py` that reads a config and wires data, model, novelty, continual tasks, poisoning, metrics, and output saving.
4. Run the tiny end-to-end smoke configuration first.
5. Fix interface mismatches and update `ARCHITECTURE.md` before changing shared assumptions.
6. Run one manageable real-data subset only after smoke passes and provenance is recorded.
7. Run GPU experiments in Colab/Lightning only after the same config and code paths work at smoke scale.

The README's future command is:

```bash
python run_pipeline.py --config configs/review2_smoke.yaml
```

It is currently a target, not a command that works in this blueprint-only repository.

## 10. Commit, hand off, and report honestly

Before a PR/session handoff:

- Review `git diff` and `git status`; ensure no raw dataset/checkpoint or personal local config is staged.
- Run the focused tests and smoke command that are available.
- Record exact Python/dependency versions, commands, seeds, input subset, commit SHA, and output paths.
- Update `PROJECT_STATUS.md` and `HANDOFF.md` with date/time, branch, completed/current work, changed files, tests, observed results, blockers, decisions, and next action.
- Update `ARCHITECTURE.md` for any approved interface change and `README.md` for actual setup/run behavior.
- Commit only your branch's scoped changes and open a focused PR for team review.

Use “not yet tested” for anything not run. Distinguish the synthetic wiring test from results on CIC-IDS2017. Never make up class counts, dataset statistics, metrics, or performance.

## Immediate next actions for this repository

1. Confirm with the team whether the current three-role ownership table is still correct given the four-person prompt.
2. Do not push commit `80018dc` with the dataset. Fetch the remote, verify whether it is unpublished, and remove the Parquet file from the local commit while preserving the local file as described above.
3. Check whether `feature/data-continual` exists; switch/create the branch according to `TEAM_WORKFLOW.md`.
4. Confirm a Python version compatible with the selected current PyTorch/scikit-learn builds and agree on dependency-manifest ownership with A.
5. Inspect the Parquet file's provenance, format/schema, labels, and class counts; verify whether it is the intended CIC-IDS2017 data.
6. Write the data card, choose a manageable reproducible subset, and decide split/task class mapping.
7. Confirm model output-head expansion and replay interfaces with A before building the continual runner.
8. Implement the data contract and tiny synthetic tests, then proceed to a tiny model integration.

## Useful links

- [Project flow](REVIEW2_PROJECT_FLOW.md)
- [Architecture and API contract](../ARCHITECTURE.md)
- [Team workflow](../TEAM_WORKFLOW.md)
- [Review-2 checklist](../REVIEW2_CHECKLIST.md)
- [PyTorch install selector](https://docs.pytorch.org/get-started/locally/)
- [scikit-learn installation guide](https://scikit-learn.org/stable/install.html)
