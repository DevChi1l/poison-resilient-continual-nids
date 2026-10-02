# Implementation Walkthrough

This is the living explanation record for implemented capstone work. It must be
updated after implementation and verification in future tasks. Existing code
that was only inspected is labeled separately from new work, and planned work
is never described as complete.

## 2026-10-02 — Confidence-based novelty baseline

Branch: `feature/unified-novelty`

Starting commit: `93492d6dc3145d0d0bcbfd481b6305cda32cd502`

### Step 1: establish the safe starting point

**What changed:** No code changed during this step. We read the repository
instructions and fetched the remote, then created `feature/unified-novelty`
from the latest verified `origin/feature/model-nids` commit. The unrelated
untracked `data/data_vis.ipynb` was left untouched.

**Why:** The model branch already combined the data pipeline and Transformer.
Starting there avoids copying or rewriting those components and makes the
novelty work directly compatible with `predict_proba()`.

**Files/functions inspected:** `AGENTS.md`, `PROJECT_CONTEXT.md`,
`TEAM_WORKFLOW.md`, `ARCHITECTURE.md`, `PROJECT_STATUS.md`, `HANDOFF.md`,
`REVIEW2_CHECKLIST.md`, `src/models/tabular_transformer.py`, and the data/model
tests. No existing source module was edited.

**Inputs, outputs, and connection:** The input to this step was the local and
remote Git graph. The output was a focused branch pointing at `93492d6`.

**Concrete example:** Remote inspection showed
`origin/feature/model-nids -> 93492d6`, while `origin/main -> 53f7a84` and
`origin/feature/novelty-poisoning -> 254a42d`. Therefore, the model branch was
the only correct foundation containing both model and data work.

**Verification:**

```bash
git status --short --branch
git log --oneline --decorate -12
git ls-remote --heads origin
git fetch --prune origin
git branch -r --contains 93492d6dc3145d0d0bcbfd481b6305cda32cd502
git switch -c feature/unified-novelty 93492d6dc3145d0d0bcbfd481b6305cda32cd502
```

Observed: the remote model branch had not advanced, the focused branch was
created, and `data/data_vis.ipynb` remained the only pre-existing untracked
path.

**Limitations and next step:** This verified source state, not runtime model
behavior. The next step was to inspect the model probability contract.

### Step 2: preserve and understand the existing foundations

**Existing code inspected, not implemented in this task:**

- `src/data/` already loads, samples, splits, and preprocesses the flow data.
  Its documented output is finite `float32` feature matrices with 54 retained
  columns and `int64` broad-class labels. Preprocessing is fitted on training
  rows only.
- `TabularTransformerClassifier.predict_proba()` returns one probability column
  per entry in `model.class_ids`.
- `TabularTransformerClassifier.predict()` takes the winning column index and
  maps it through `class_ids` to recover the external class ID.

**Why this matters in the NIDS pipeline:** Novelty detection consumes the
classifier's uncertainty, so it must preserve the exact model/data interface.
It must not retrain the model, reinterpret features, or treat column number as
the class label.

**Inputs and outputs:** The model consumes preprocessed flow rows and produces
an `(n_samples, n_known_classes)` probability matrix. The novelty component
consumes that matrix. For a model with `class_ids=(10, 30, 90)`, probability
column 1 means external class 30, not class 1.

**Verification:** Source inspection confirmed that `predict_proba()` applies
softmax and that `predict()` indexes `np.asarray(self.class_ids)` using the
winning probability-column positions.

**Limitations and next step:** Real-data model training was not run. PyTorch is
not installed in the current environment. This did not block a NumPy-only
novelty component or synthetic tests.

### Step 3: implement confidence thresholding

**What we implemented:**

- `src/novelty/confidence.py`
  - `detect_unknown(probabilities, threshold)`
  - `summarize_predictions(probabilities, class_ids, threshold)`
  - immutable `NoveltyResult`
- `src/novelty/__init__.py` exposes those public names.
- `tests/test_novelty.py` contains nine synthetic component tests.

**Why it is needed:** A closed-set classifier always chooses one known output
class, even for an unfamiliar flow. The baseline adds an `UNKNOWN` decision for
low-confidence rows so later evaluation can measure whether held-out attacks
are rejected.

**How Transformer probabilities become confidence scores:** Softmax in
`predict_proba()` produces a probability distribution across known output
classes for each row. We use the largest value in a row as that row's confidence
because it is the probability assigned to the model's selected class. This is
simple, deterministic, and matches the frozen architecture contract.

**How the threshold works:**

```python
confidence = np.max(probabilities, axis=1)
unknown = confidence < threshold
```

The comparison is strictly `<`. Confidence equal to the threshold is known.
This makes the boundary explicit and matches the documented rule; using `<=`
would silently change which samples are rejected.

**Concrete example:**

```python
import numpy as np
from src.novelty import detect_unknown, summarize_predictions

probabilities = np.array([
    [0.10, 0.70, 0.20],
    [0.45, 0.25, 0.30],
])
class_ids = [10, 30, 90]

unknown = detect_unknown(probabilities, threshold=0.50)
# array([False, True])

details = summarize_predictions(probabilities, class_ids, threshold=0.50)
# details.predicted_class_ids -> array([30, 10])
# details.maximum_probabilities -> array([0.70, 0.45])
# details.is_unknown -> array([False, True])
```

The first winning column is position 1, which maps to external class ID 30.
The second winning column is position 0, which maps to external class ID 10,
but its confidence is below 0.50 and is therefore marked unknown.

**Validation and non-mutation:** The code requires a two-dimensional real
numeric matrix with at least one class column. Every value must be finite and
within `[0, 1]`, and each row must sum approximately to one. The threshold must
be a finite real scalar in `[0, 1]`. The helper additionally requires unique
integer `class_ids` with exactly one ID per probability column. Invalid input is
rejected rather than normalized because silent repair could conceal a broken
model/integration boundary. The functions only read the probability array;
non-mutation matters because the same output may also feed classification
metrics or an audit record.

An empty matrix shaped `(0, n_classes)` is valid and returns one-dimensional
empty arrays. `(n_samples, 0)` is invalid because confidence cannot be defined
without a class column.

### Step 4: verify behavior

**Exact commands and observed outcomes:**

```bash
python3 -m unittest tests.test_novelty -v
```

Observed: 9 passed, 0 failed, 0 skipped. Cases cover known/unknown decisions,
equality at the threshold, threshold endpoints, empty batches, malformed
probabilities, invalid thresholds/class IDs, non-contiguous external IDs, and
input preservation.

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Observed: 16 passed, 0 failed, 1 skipped. The skip was the opt-in real-data
model/data smoke test. It remains honestly reported as skipped; no real-data
training claim was made.

```bash
python3 -m compileall -q src/novelty tests/test_novelty.py
```

Observed: passed with no output.

```bash
python3 -m pytest -q
```

Observed: the command did not start the suite because `/usr/bin/python3`
reported `No module named pytest`. We did not install pytest or the GPU stack
for this isolated task.

Environment inspection found NumPy, pandas, and PyArrow available; PyTorch and
pytest were not installed. The novelty tests require only NumPy and do not load
the dataset.

```bash
git diff --check
```

Observed: passed with no whitespace errors on the complete implementation and
documentation diff.

### Step 5: unify ownership and persistent context

**What changed:** `AGENTS.md`, `TEAM_WORKFLOW.md`, `ARCHITECTURE.md`, `README.md`,
`PROJECT_CONTEXT.md`, `PROJECT_STATUS.md`, `HANDOFF.md`, and
`REVIEW2_CHECKLIST.md`, and relevant source/script/project-flow documentation
now record A+B+C as one owner and distinguish completed components from pending
integration. This walkthrough was added as the detailed explanation record.

**Why:** The prior ownership tables would incorrectly cause future work to wait
for or reassign tasks to former separate owners. Stale “implementation has not
started” text also contradicted the checked-in data and model code.

**Connection:** The unified team now owns both component directories and shared
integration files, while focused branches and documented interfaces continue to
protect stable work.

**Limitations and next step:** Documentation does not make the end-to-end
prototype complete. The next implementation task is deterministic,
configurable training-label flipping. Continual task construction, replay,
mitigation, evaluation, and integration remain pending team responsibilities.

## Why this is only a baseline

Maximum softmax confidence can be low for difficult known samples and high for
unfamiliar samples. Therefore, the implemented rule demonstrates the interface
and produces a reproducible unknown mask, but it does not establish
unknown-attack detection performance. A valid experiment must train without one
or more selected classes, use those held-out classes only as unknown evaluation
examples, retain known-class test examples, choose or calibrate the threshold
without using the final test labels, and then report recorded novelty metrics.
No such protocol or metric was run in this task.

## How to explain this in Review 2

The Transformer gives a probability for every attack class it already knows.
We take the largest probability as its confidence in the chosen class. If that
confidence is below a configurable threshold, we flag the flow as unknown;
confidence exactly on the threshold stays known. This is a small, transparent
baseline that connects cleanly to the model, but we will only claim detection
performance after testing it against attack classes deliberately excluded from
training.

## Likely faculty questions and accurate answers

**Why use maximum probability?** It is the confidence attached to the class the
model would predict, requires no extra model, and is the agreed Review-2
baseline. It is not claimed as a sophisticated novelty detector.

**Why is equality considered known?** The architecture defines unknown as
strictly below the threshold. A strict boundary is deterministic and avoids
ambiguity at the configured cutoff.

**Does probability column 1 mean class 1?** Not necessarily. It means
`model.class_ids[1]`. The helper requires the complete `class_ids` mapping and
tests non-contiguous IDs such as 10, 30, and 90.

**How did you choose the threshold?** We did not calibrate or claim a best
threshold in this component task. It is configurable. A later experiment must
select it using a documented validation protocol rather than the final test
labels.

**Have you detected zero-day or unknown attacks?** No performance claim has
been established. We implemented and tested the decision rule with synthetic
probabilities. Held-out unknown-class evaluation is still required.

**Why validate probabilities instead of normalizing them?** Invalid sums,
NaNs, infinities, or out-of-range values indicate an upstream error. Silently
normalizing them would hide that error and make results harder to audit.

**Did you train the Transformer on the real dataset?** No. The model source and
integration test exist, but real-data training remains unverified and the
current environment lacks PyTorch. The real-data test is reported as skipped.

**What comes next?** Deterministic label flipping on training labels only,
followed later by continual tasks, replay, mitigation, evaluation, and the
end-to-end runner.

---

## 2026-10-02 — Kaggle GPU smoke notebook preparation

Notebook: `notebooks/kaggle_review2_smoke.ipynb`

Published source revision pinned inside the notebook:
`22e797660307bfed62b739d61b5e4c05bcb0a187` on
`feature/unified-novelty`. The notebook commit itself cannot be its own source
pin; the pinned revision contains the existing data, model, and novelty APIs
that the notebook will execute.

### Step 1: verify the foundation and define scope

**What changed and why:** We fetched `origin` and verified that local and
published `feature/unified-novelty` matched the full SHA above. `main` is behind
and does not contain the combined implementation. We chose the published
unified revision to make the Kaggle code checkout repeatable. This is a smoke
test of existing components, not a continual-learning or poisoning experiment.

**Files and connection:** We inspected `src/data/{sampling,splitting,preprocessing}.py`,
`src/models/tabular_transformer.py`, `src/novelty/confidence.py`, and their
public package exports. The notebook imports those APIs directly and does not
change them. The untracked `data/data_vis.ipynb` was preserved.

**Verification:** `git fetch --prune origin`, `git status --short --branch`,
`git log --oneline --decorate -8`, and `git ls-remote --heads origin` showed
the unified remote/local branch at the same SHA. This verifies the source
revision, not Kaggle execution.

### Step 2: set up and inspect Kaggle

**What the notebook does:** Its opening Markdown explains how to import the
`.ipynb`, enable Internet and GPU, and attach the private combined flow Parquet
file as a Kaggle input. The first code cell clones the published branch,
checks out the exact source SHA in detached mode, asserts it, and prints it.
The second stage records Python, NumPy, pandas, PyArrow, and PyTorch versions,
CUDA availability, CUDA build, GPU name, and selected device. Small import and
array checks catch basic compatibility problems. There is no package install
command, so Kaggle's supplied CUDA-enabled PyTorch remains intact.

**Why and inputs/outputs:** The input is the public source repository and the
Kaggle runtime; the outputs are an exact code checkout and an environment
record for the manifest. GPU is required unless the user explicitly changes
`ALLOW_CPU_FALLBACK` to `True`.

**Concrete example:** If CUDA is unavailable with the default setting, the
notebook raises a message asking for a Kaggle GPU. It does not silently run a
CPU experiment.

### Step 3: check the model before reading data

**What the notebook does:** It creates 16 seeded synthetic rows with four
features and two external class IDs, fits a tiny instance for one epoch, and
checks finite losses, prediction/probability shapes, unit probability sums,
and `class_ids` mapping.

**Why and connection:** This isolates model/runtime compatibility before a
full Parquet scan. The synthetic run is labeled functional verification only;
it does not provide NIDS evidence.

### Step 4: locate and prepare the combined flow collection

**What the notebook does:** It finds `cic-collection.parquet` under
`/kaggle/input`, with an editable `PARQUET_PATH`. Zero or multiple automatic
matches stop with a clear error. It invokes the existing
`sample_class_balanced_indices(..., per_class_limit=64, seed=42)`,
`stratified_split_indices(..., seed=42)`, and `prepare_sampled_dataset(...)`.
It prints the partition shapes, broad class mapping, available source counts,
finite-value checks, and stage timings.

**Why and connection:** The sampler returns original Parquet row indices while
holding at most 64 per class in its reservoirs. It still scans the complete
file; preprocessing scans to materialize selected rows and fits medians, means,
and scales only on training rows. The resulting 54-column `float32` matrices
and `int64` broad labels connect directly to the existing classifier.

**Concrete example:** For eight broad classes, 64 rows each gives 512 selected
rows before splitting. The exact train/validation/test shapes are printed by
the Kaggle run and are not prefilled as observed results here.

### Step 5: train, evaluate, and save what actually happened

**What the notebook does:** It constructs the repository classifier with
`hidden_dim=16`, `num_heads=2`, `num_layers=1`, `mlp_dim=32`, `batch_size=32`,
and one epoch. The model uses GPU by default. It checks finite history values,
test prediction and probability shapes, probability sums, and the mapping
from winning probability column through `model.class_ids`. The notebook
computes observed test accuracy, macro-F1, per-class precision/recall/F1, and
a confusion matrix using the actual test labels and predictions. It calls
`detect_unknown` with an editable threshold and reports flag counts only.

**Why and outputs:** This checks that the full data-to-model path runs and
produces auditable outputs. All broad classes are represented during training,
so a confidence flag cannot be interpreted as an unknown-attack detection
success. A held-out unknown-class protocol remains necessary.

**Artifact connection:** A timestamped `/kaggle/working/review2_smoke_*`
folder will contain `transformer.pt`, `row_indices.npz`,
`preprocessing.json`, `metrics.json`, `training_history.json`, and
`manifest.json`. The indices file stores both sampled original rows and split
positions/original rows. The preprocessor file stores feature order, fitted
imputation values, means, scales, and class names. The manifest records the
source SHA, environment, input path/size/SHA-256, seeds, class order, config,
threshold, timings, observed metrics, history, and artifact names. A final
cell reloads the checkpoint and requires identical test predictions before
marking reload verification true in the manifest.

### Step 6: local verification and remaining limits

**Exact commands run locally:**

```bash
python3 - <<'PY'
import ast
import json
from pathlib import Path
notebook = json.loads(Path('notebooks/kaggle_review2_smoke.ipynb').read_text())
assert notebook['nbformat'] == 4
for index, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        assert cell['execution_count'] is None and cell['outputs'] == []
        ast.parse(''.join(cell['source']), filename=f'cell-{index}')
        assert notebook['cells'][index - 1]['cell_type'] == 'markdown'
print(len(notebook['cells']))
PY
python3 -m unittest tests.test_novelty -q
python3 -m unittest discover -s tests -p 'test_*.py' -q
git diff --check
```

**Observed:** Notebook JSON and all code cells parsed; every code cell had
preceding Markdown and no saved execution output. The notebook has 19 cells,
including 9 code cells. Nine focused novelty tests passed. Repository
discovery ran 17 tests: 16 passed and the opt-in real-data model test was
skipped. The diff whitespace check passed. `nbformat` and PyTorch are not
installed locally, so the notebook was not executed here. The Kaggle GPU,
dataset attachment, training history, metrics, checkpoint, and reload status
remain unverified until the user runs it.

**Next step:** Run it on Kaggle and review the downloaded manifest, metrics,
history, final printed summary, and any errors before choosing the next
implementation prompt. Deterministic training-label flipping remains planned
but is outside this notebook phase.

### How to explain this in Review 2

We pinned the exact published code revision, recorded the Kaggle runtime, and
tested the model on a tiny synthetic batch before touching the real input. The
real smoke path then selects a seeded balanced subset of the combined flow
collection, fits preprocessing only on training rows, runs one small GPU
training epoch, and saves every input choice and observed output needed to
reproduce or inspect the run. Its classification metrics describe only this
small known-class smoke run. Low-confidence flag counts demonstrate the
novelty interface; they do not measure detection of unseen attacks.

---

## 2026-10-03 — Review completed smoke and prepare clean baseline training

This section updates the earlier planning record: the smoke notebook has now
run on Kaggle. The larger clean baseline notebook described below has **not**
run. Developers A+B+C remain one owner of the model, data, novelty, and
integration work.

### Step 1: inspect the supplied smoke evidence

**What we reviewed:** The user attached `data/manifest.json`,
`data/metrics.json`, `data/training_history.json`, and
`data/notebook-smoke.ipynb`. These are untracked local execution artifacts;
they were read for evidence and were not added to Git. Their JSON values agree,
and the executed notebook's code cells have no error outputs.

**Why it matters:** This is the first observed real-data Transformer fit in the
project. It confirms the data-to-model wiring on Kaggle CUDA and demonstrates
checkpoint reload, while providing a concrete reason to run a longer clean
baseline before judging model quality.

**Observed inputs and outputs:**

- Source revision: `22e797660307bfed62b739d61b5e4c05bcb0a187`.
- Combined flow input SHA-256:
  `666af53c788c79312b421c607289cb555c2335ae51ba39b2700a357595fa8dba`.
  The Parquet file still has unverified per-row source provenance.
- Kaggle Tesla T4, Python 3.12.13, PyTorch 2.10.0+cu128, CUDA used.
- Seed 42, 64 samples per broad class, train `(360, 54)`, validation
  `(80, 54)`, test `(72, 54)`.
- One epoch: train loss `2.1014001899295383`, validation loss
  `1.9477939367294312`, validation accuracy `0.275`.
- Test accuracy `0.25`, macro-F1 `0.1915376915376915`; checkpoint reload
  predictions were identical.
- Confidence threshold `0.5` flagged all 72 **known-class** test rows.
  This counts low-confidence predictions and is not unknown-attack detection
  performance. Lowering the threshold to improve this count would not validate
  the novelty method.

**Why accuracy was preliminary:** The fit saw only 360 training rows total,
one epoch, and the test had nine rows per class. The artifacts establish a
working smoke path, not a final accuracy estimate or a comparison of training
methods. The 2,000-per-class clean baseline increases the training subset and
allows validation-loss model selection without consulting test labels.

**Availability decision:** The smoke manifest counted at least 2,255 rows in
every broad class, so 2,000/class is feasible for the same checksum. The new
notebook verifies the checksum and class availability again at run time.

### Step 2: add visible training progress to the existing model

**What we implemented:** `TabularTransformerClassifier.fit` gained the
optional `verbose=False` argument in `src/models/tabular_transformer.py`.
With `verbose=True`, it prints completed-epoch train loss, validation loss,
and validation accuracy, plus early-stop and best-state restoration messages.
The default stays quiet for existing callers. The existing validation-loss
comparison, patience count, best-state copy, and final restoration were not
changed. The public interface was recorded in `ARCHITECTURE.md` and
`src/models/README.md`; the PyTorch model test now checks progress text.

**Why:** A run lasting up to 20 epochs needs visible progress in Kaggle so the
team can tell whether training is advancing and which epoch was retained.
This is observability, not a new training algorithm.

**Concrete example:** After epoch 3, a verbose run prints the observed train
loss, validation loss, and validation accuracy for epoch 3. If validation
loss later fails to improve for five epochs, the model stops and restores the
state from the lowest-loss validation epoch before `fit` returns. The example
does not assert any expected numeric loss for the pending baseline.

**Source pin:** The progress change was committed and normally pushed as
`68135601a3ddd95832c57bb602f210f1a0e18a43`. The clean notebook checks
out this full published SHA, so every repository API it calls is present.

### Step 3: prepare the clean Kaggle notebook

**What we implemented:** `notebooks/kaggle_review2_clean_baseline.ipynb` uses
the existing sampler, splitter, preprocessor, classifier, and confidence
baseline. It retains the completed smoke notebook unchanged. Each code stage
has explanatory Markdown.

**Inputs and connection:** The notebook imports a private attached
`cic-collection.parquet`, requires the smoke input SHA-256, and uses 2,000
rows per broad class with sample and split seed 42. The existing stratified
split gives 70/15/15 partitions; the existing preprocessor fits medians,
means, and scales only on training rows. The sampler still scans the full
Parquet label column, and materialization scans bounded batches to find the
selected rows. No processed dataset is committed.

**Model and selection:** A newly constructed model uses `hidden_dim=64`,
`num_heads=4`, `num_layers=2`, `mlp_dim=128`, batch size 128, learning rate
0.001, at most 20 epochs, patience 5, CUDA required, mixed precision off,
and seed 42. The earlier smoke checkpoint is never loaded. `fit` sees train
and validation partitions only, prints each epoch, and returns with its
lowest-validation-loss state restored. The notebook plots observed training
and validation loss and validation accuracy, marking the selected epoch.

**Concrete sample-size example:** Eight classes times 2,000 rows gives 16,000
selected rows if every class passes availability checks. The expected
70/15/15 split would then be 11,200 train, 2,400 validation, and 2,400 test
rows. These are arithmetic expectations, not observed clean-baseline outputs;
the notebook prints and records actual shapes when run.

**Test and outputs:** Only after `fit` completes and restores the best state
does the notebook call `predict` and `predict_proba` on test features. It
checks shape, finite/unit-sum probabilities, and mapping from probability
column to `model.class_ids`. It computes accuracy, macro-F1, per-class
precision/recall/F1, and a labeled confusion matrix. Threshold 0.5 is kept
as a known-class confidence diagnostic; flag counts are reported, not novelty
performance. Test results are for reporting this fixed configuration and must
not be used to tune it.

The timestamped `/kaggle/working/review2_clean_baseline_*` folder will hold
`manifest.json`, `training_history.json`, `metrics.json`,
`training_curves.png`, `confusion_matrix.png`, `row_indices.npz`,
`preprocessing.json`, and `transformer_best_validation.pt`. The manifest
records actual environment versions, source SHA, input path/size/checksum,
sampling/split seeds, configuration, model-selection epoch, timings, metrics,
artifact names, and reload verification. The final cell reloads the saved
checkpoint and requires identical test predictions before setting the reload
flag to true.

### Step 4: local verification and limits

**Commands and observed outcomes:**

```bash
python3 -m compileall -q src/models tests/models
python3 -m unittest discover -s tests -p 'test_*.py' -q
git diff --check
```

Compilation and whitespace checks passed. `unittest` discovery ran 17 tests:
16 passed and the opt-in real-data model test was skipped. The PyTorch
progress test is written in the existing `pytest` model suite but could not
execute locally because PyTorch and pytest are absent. The earlier Kaggle
smoke success must not be described as a local model-test pass for this new
progress change.

Notebook validation parsed the `.ipynb` JSON and all Python code cells,
confirmed Markdown immediately before every code cell, and found no saved
execution outputs: 21 cells, 10 code cells. No clean-baseline GPU training,
metric, plot, checkpoint, or reload result has been generated yet.

### How to explain this in Review 2

The one-epoch T4 smoke showed that our checked-in data, model, and checkpoint
interfaces work together, but it was too small to judge classification
quality. The clean baseline keeps the dataset, class mapping, seeds, and
train-only preprocessing fixed while giving a larger balanced sample and up
to 20 training epochs. Validation loss chooses the retained model before we
look at the test partition. We will report the actual per-class results and
confusion matrix after Kaggle runs it. Confidence flags remain diagnostics
until we design a held-out unknown-class evaluation.
