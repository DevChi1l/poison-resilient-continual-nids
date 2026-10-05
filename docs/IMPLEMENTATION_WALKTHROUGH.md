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

---

## 2026-10-03: observed clean baseline and static label-flip preparation

### Step 1: review actual clean-baseline evidence

**Inspected, not implemented here:** the attached untracked clean manifest,
metrics, history, executed notebook, loss curves, and confusion-matrix image.
Its source was `68135601a3ddd95832c57bb602f210f1a0e18a43`, input SHA-256
`666af53c788c79312b421c607289cb555c2335ae51ba39b2700a357595fa8dba`,
on Tesla T4 with Python 3.12.13 and PyTorch 2.10.0+cu128. The 2,000-per-class
balanced sample yielded 11,200 train, 2,400 validation, and 2,400 test rows,
each with 54 retained features. Training ran 20 epochs; minimum validation
loss selected epoch 17. The observed test accuracy was
`0.8833333333333333` and macro-F1 `0.878016638838182`. Training took
`13.00852884599999` seconds, and saved-checkpoint reload predictions matched.
The notebook had no error outputs. Benign recall was `0.44`: 168/300 true
Benign flows were called attacks, including 138 called Infiltration. Thus
Benign false-positive rate is `168/300 = 0.56`. This specific weakness makes
Benign-targeted corruption worth comparing, but it does not predict an attack
outcome. This is balanced-subset evidence, not full-dataset or deployment
performance. The clean notebook and artifacts were not edited or committed.

### Step 2: implement deterministic attack mechanics

**New files/functions:** `src/poisoning/label_flip.py` defines
`apply_label_flip` and `PoisoningResult`; `src/poisoning/__init__.py` exports
them. Inputs are one-dimensional integer training labels, a rate in `[0,1]`,
a nonnegative seed, valid class IDs, and optionally original dataset row IDs.
Mode `random` samples from every training row and gives each selected label a
uniformly chosen *different* allowed class. Mode `targeted` samples only
specified source-class rows not already in the explicit target class and
assigns the target, initially Benign. Rows are sampled without replacement.
The count is `floor(rate * eligible_count + 0.5)`, including half-up ties.

**Why and how it connects:** static label corruption lets us quantify a
controlled training-data integrity failure without changing network-flow
features, model code, or clean validation/test labels. The returned copied
`poisoned_labels` feed the existing model's `fit(X_train, y_train, X_val,
y_val)`; the original labels and selected original row indices remain in the
audit. Random rate divides by all training rows, targeted rate by eligible
source rows, so the result separately reports eligible and total fractions.
For a ten-row training array with four eligible source rows, rate 0.25 changes
three random rows (`floor(2.5+0.5)`) but only one targeted row. Calling both
"25% of training" would be false.

**Validation and preservation:** malformed rates/seeds, noninteger labels,
duplicate or missing classes, invalid target/source choices, and misaligned
original row IDs raise errors. Both the input labels and original-row array
are left untouched; even zero rate returns a separate copied label array.
This matters because clean labels are needed for fair comparison and any
accidental in-place mutation would contaminate later conditions.

### Step 3: centralize classification and attack diagnostics

**New files/functions:** `src/evaluation/classification.py` exports
`classification_metrics`. It takes true/predicted external class IDs and
optional explicit class mapping, names, Benign ID, source IDs, and target ID;
it returns a JSON-serializable dictionary. Confusion counts, accuracy,
macro-F1, per-class precision/recall/F1, Benign false-positive rate, and
source-to-target misclassification are computed once for all conditions.
For example, if two of four true attack-source rows are predicted Benign,
the source-to-Benign error rate is 0.5 even for a clean model; the poisoned
condition's *difference* from its new 0% control is the meaningful comparison.
No per-row causal claim follows from that difference. Probability columns are
still aligned to `model.class_ids`, not assumed to equal external IDs.

### Step 4: prepare the Kaggle comparison notebook

**New file:** `notebooks/kaggle_review2_label_flip.ipynb`. It clones the
published source commit `316b817b40f4008af128489ac958d3b232b56049`,
checks CUDA and package versions without replacing Kaggle's PyTorch, locates
the attached private Parquet, and requires the reviewed input checksum. It
uses existing `src.data` functions to sample once at 2,000 per broad class
with seed 42, stratify 70/15/15 with seed 42, and fit preprocessing on train
only. Bounded sampling still scans the large input. The class mapping is
checked before deriving Benign target and attack-source IDs.

The conditions are a newly trained 0% control plus random and targeted
flips at 5%, 10%, and 20%. Each fresh Transformer uses 64 hidden units,
four heads, two layers, MLP width 128, batch 128, learning rate 0.001,
up to 20 epochs, patience five, CUDA, no mixed precision, and seed 42.
Initial state hashes are required to match. Poisoned labels replace only
training targets. The existing `fit` restores minimum **clean validation**
loss weights; that assumes a trusted validation partition and is not an
implemented defense. Test features/labels are used only after model selection.
Every condition writes its attack counts/denominators, changed training and
original dataset row indices, history, metrics, timings, checkpoint, and
reload check to its own folder. A shared manifest records the Git SHA,
environment, input checksum, split rows, preprocessing state, and config.
The comparison reports deltas from the newly run 0% control and per-class
recall/F1. All-known confidence flag counts remain diagnostics only.

### Step 5: verification, limitations, and next step

The focused synthetic command
`python3 -m unittest tests.test_poisoning tests.test_evaluation -v` passed
six tests, covering reproducibility, half-up counts, valid random
replacements, source-only targeted eligibility, zero/full rates, invalid
inputs, input preservation, noncontiguous IDs, and diagnostic metrics.
Notebook JSON parsed and every code cell passed `ast.parse`: 15 cells (7
code), with Markdown before each stage and no saved execution outputs.
`python3 -m unittest discover -s tests -p 'test_*.py' -q` ran 23 tests:
22 passed, 1 opt-in real-data model test skipped. `python3 -m compileall -q
src/poisoning src/evaluation tests/test_poisoning.py tests/test_evaluation.py`
and `git diff --check` passed. The local environment cannot run the
CUDA/private-dataset comparison; no poisoning result has been observed.
Kaggle output must be reviewed before
claiming any effect. Static label corruption is not replay poisoning or a
timing backdoor. Feature triggers need feature-semantic/bound inspection first.
Next implementation after the Kaggle review: clean continual tasks and
balanced replay, with mitigation still our responsibility.

### How to explain this in Review 2

We first trained a clean Transformer and measured an 88.33% balanced-subset
accuracy, but also a 56% false-positive rate for Benign. Now we have prepared
a controlled test of whether wrong training labels change that behavior.
Every run uses the same sampled flows, splits, preprocessing, architecture,
and starting weights; only selected training labels differ. Random flips
send each selected label to another class, while targeted flips make selected
attack labels Benign. We compare each to a fresh clean control, show both
actual row counts and rates, and do not claim results until Kaggle runs it.

Likely faculty questions:

- **Are 10% random and 10% targeted the same number of changes?** No. Random
  uses all training rows; targeted uses only eligible attack-source rows.
- **Is clean validation loss a defense?** No. It assumes validation labels
  are trusted and chooses a checkpoint under that assumption.
- **Does an attack-to-Benign error prove poisoning caused it?** No. The clean
  model may make the same error; we compare aggregate rates with the control.
- **Is this replay poisoning or unknown-attack detection?** Neither. Labels
  are corrupted once before ordinary training, and test classes are known.

---

## 2026-10-03: observed poisoning comparison and clean continual/replay preparation

### Step 1: inspect the completed static poisoning run

**Evidence inspected, not new training:** the untracked
`data/poison/manifest.json`, `comparison.json`, and executed `poison.ipynb`.
The notebook has no error outputs. Its source SHA was
`316b817b40f4008af128489ac958d3b232b56049`, input checksum matched the
reviewed combined flow collection, all seven checkpoint reloads passed, and
all initial-weight hashes matched. The 0% control repeated accuracy
`0.8833333333333333`, macro-F1 `0.878016638838182`, and Benign FPR `0.56`.
Random 20% changed 2,240 of all 11,200 training rows: accuracy `0.87`,
macro-F1 `0.8590190796024948`, Benign FPR `0.67`. Targeted 20% changed
1,960 of 9,800 eligible attack-source rows, or 17.5% of all training rows:
accuracy `0.8533333333333334`, macro-F1 `0.8514854740614037`, and
attack-to-Benign rate `0.06571428571428571` versus clean
`0.025238095238095237`. These nominal 20% settings have *different
denominators*. The effects were mixed: random 10% accuracy (`0.88375`) and
targeted 10% macro-F1 (`0.8817804870790201`) slightly exceeded control.
One seed and one balanced subset cannot establish that poisoning always
degrades metrics. No attachment was committed.

### Step 2: construct global-ID tasks without future-data fitting

**New code:** `src/continual_learning/tasks.py` defines
`prepare_two_task_dataset`, `TaskPartitions`, and `ContinualTasks`.
Inputs are `SelectedRows` from the existing
`src.data.materialize_selected_rows` and one `StratifiedSplit` from the
existing seed-42 splitter. The builder calls the existing
`encode_broad_labels` once with the canonical eight-class order. Task 1
keeps IDs 0–4 (Benign, DDoS, DoS, Botnet, Bruteforce); Task 2 keeps IDs
5–7 (Infiltration, Webattack, Portscan). Neither task is re-encoded from
zero. It validates that the selected raw row IDs and split partitions are
unique, complete, disjoint, and contain every required class.

**Why/how:** a class-incremental model needs stable output IDs; re-encoding
Task 2 as 0–2 would confuse new attacks with old classes. More importantly,
the existing `prepare_sampled_dataset` fits on *all* training rows, including
future Task 2 rows. For this protocol the builder instead calls
`fit_preprocessor` only on Task 1 training features, then applies its frozen
imputation/scaling to both tasks' train, validation, and test features. If a
future Task 2 feature is an outlier of 10,000 while Task 1 training values
are below 100, that outlier cannot alter the fitted Task 1 mean. Synthetic
tests verify this exactly. Each output partition has finite-model-ready
features, global `int64` labels, and original Parquet row indices for audit.
The bounded sample still requires a scan of the Parquet label column and
materialization of selected flows.

### Step 3: select a clean balanced memory

**New code:** `src/continual_learning/replay.py` defines
`select_balanced_replay` and `ReplaySelection`. It accepts only the prepared
Task 1 **training** partition, global old-class IDs, `per_class=100`, seed
42, and forbidden validation/test raw row IDs. It samples without replacement
and returns copied features, labels, source training positions, per-class
counts, and original row IDs. `as_fit_replay()` returns fresh copies in the
existing model's `(X_replay, y_replay)` contract. Overlap with forbidden
rows, insufficient class size, invalid IDs, and duplicates are rejected.

**Concrete example:** five old classes times 100 gives 500 stored rows. The
sequential arm trains on 4,200 new-class rows per epoch; the replay arm
trains on those same 4,200 plus 500 old rows. At batch size 128, that is
33 versus 37 optimizer steps per epoch. Because work differs, any score
difference cannot be attributed solely to which examples were remembered.
The notebook records actual epochs and total optimizer steps, including
early stopping effects.

### Step 4: define signed forgetting without dropping new-class errors

**New code:** `src/evaluation/continual.py` defines `forgetting_metrics`.
Inputs are the *same* old-class test truth array and predictions before/after
Task 2, plus explicit old and all-class IDs. It uses the shared classifier
evaluator, then averages F1 over the same five old classes at both times.
If an old-class row is predicted as a new class after Task 2, it counts as
an error. Outputs include before/after old accuracy and macro-F1, signed
after-minus-before changes, and before-minus-after forgetting. Negative
forgetting is an improvement, not truncated to zero. For example, if old
accuracy goes from 0.5 to 0.75, accuracy forgetting is `-0.25`.

### Step 5: prepare the controlled Kaggle training comparison

**New notebook:** `notebooks/kaggle_review2_continual_replay.ipynb` pins
published source `b2a0afc697272a9d093780057f38c63ee19340b0` and requires
the previously reviewed input checksum. It samples 2,000 per broad class,
splits once at 70/15/15 with seed 42, materializes rows once, and builds
tasks with Task-1-only preprocessing. It selects 100 unique old training
rows per class for replay and records their original row IDs.

A fresh five-output Transformer trains on Task 1 train/validation only;
`class_ids` must remain 0–4. No eight-class static checkpoint or future
validation label is supplied. Its best clean-validation-loss checkpoint is
saved. Both Task 2 arms independently reload that *same* checkpoint. Each
load resets the model seed before `add_classes(5,6,7)`; the complete expanded
weight hashes must match. The notebook also resets training RNG before each
Task 2 fit. Both arms use the same Task 2 training rows, architecture,
learning rate, batch size, maximum epochs, and patience. Sequential receives
no replay; replay receives the 500-row `(X, y)` buffer. Both use the same
clean validation pool of all classes seen by Task 2. This explicitly assumes
trusted old/new validation data; it is not mitigation. The test partition
does not control checkpoint selection.

After fitting, the notebook evaluates Task 1 test before Task 2, Task 1
test after each arm, Task 2 test after each arm, and combined test after
each arm. It records accuracy, correctly focused old/new macro-F1, per-class
recall/F1, Benign false-positive rate, signed forgetting, train/validation
histories, epochs, training rows, optimizer steps, timings, curves, and
checkpoint reload checks. The timestamped `/kaggle/working/` folder stores
`manifest.json`, `summary.json`, Task 1 and arm metrics/history/checkpoints,
`row_indices.npz`, `replay_indices.npz`, copied `replay_buffer.npz`,
`preprocessing.json`, and
`training_curves.png`. The observed values will be filled in by Kaggle; no
continual-learning result exists yet.

### Step 6: verification and boundaries

`python3 -m unittest tests.test_continual_learning -v` passed four synthetic
tests for task membership/global IDs, split disjointness, frozen Task 1 fit,
balanced deterministic replay/copying/exclusions, and signed forgetting with
new-class prediction errors. The full local `unittest` suite ran 27 tests:
26 passed, one opt-in real-data model check skipped. `compileall`, notebook
JSON/code syntax and Markdown-stage checks, and `git diff --check` passed.
The notebook has 23 cells (11 code), all without execution outputs. Local
PyTorch/CUDA and the private Kaggle input are unavailable, so no local model
fit or GPU run is claimed. This is clean continual learning only; replay
poisoning, mitigation, timing backdoors, and full-data streaming await
execution review.

### How to explain this in Review 2

We teach the model five broad classes first, then introduce three new ones.
One update learns only from the new classes; the other also rehearses 100
saved training flows from each old class. Both start from the same Task 1
checkpoint, and neither gets future examples when preprocessing is fitted.
We measure old-class scores before and after, new-class scores after, and
the extra training work replay requires. The notebook is prepared; we will
make no claim about forgetting until its Kaggle outputs are reviewed.

Likely faculty questions:

- **Why not fit preprocessing on all training rows?** That would let future
  Task 2 data influence Task 1, leaking information across the task boundary.
- **Why retain IDs 0–7?** Class-incremental outputs must represent the same
  class across tasks; Task 2 ID 5 must not be remapped to ID 0 (Benign).
- **Is replay fair if it adds optimizer steps?** We hold model, Task 2 data,
  initialization, and validation fixed but record extra rows/steps; it is a
  practical replay comparison, not an equal-compute causal ablation.
- **What if forgetting is negative?** It means old-class performance
  improved. We retain its sign.
- **Is this replay poisoning?** No. The memory contains clean Task 1 training
  exemplars. Replay poisoning and mitigation are later phases.

---

## 2026-10-03: observed clean continual result and prepared replay-label gate

### Step 1: inspect the clean continual evidence before changing anything

**Inspected, not trained in this session:** untracked
`data/continual/continual.ipynb`, `manifest.json`, `summary.json`,
`replay_buffer.npz`, and `training_curves.png`. The executed notebook has no
error outputs. The replay array is `(500, 54)` `float32` with 500 unique raw
row IDs and 100 `int64` labels per old global class. The source revision is
`b2a0afc697272a9d093780057f38c63ee19340b0`; its input checksum matches
the reviewed combined collection. The runtime was Tesla T4, Python 3.13.15,
PyTorch 2.11.0+cu128, NumPy 2.1.3, PyArrow 23.0.1. Earlier static-poisoning
artifacts reported Python 3.12.13, PyTorch 2.10.0+cu128, NumPy 2.0.2,
PyArrow 24.0.0, so exact weights should not be assumed reproducible across
these runtimes.

Observed Task 1 old-class accuracy before Task 2 was
`0.9686666666666667`. Sequential fine-tuning left old accuracy `0`, new
accuracy `0.9377777777777778`, combined accuracy
`0.3516666666666667`, and Benign FPR `1.0`. Clean class-balanced replay
left old accuracy `0.6826666666666666`, new accuracy
`0.9433333333333334`, combined accuracy `0.7804166666666666`, and Benign
FPR `0.9366666666666666`. The sequential/replay arms used 264/444 optimizer
steps (8/12 epochs). The plotted clean validation curves and manifest agree
that model selection occurred separately for the two arms. Replay improved
old retention relative to sequential, but old accuracy still lost `0.286`
from Task 1 and Benign errors remained severe. This single-seed
balanced-subset comparison is neither equal-compute nor deployment evidence.
The attached files include the replay buffer but no Task 1 checkpoint; we
do not fabricate an equivalent local checkpoint.

### Step 2: implement the frozen-teacher consistency baseline

**New code:** `src/mitigation/label_consistency.py` exports
`label_inconsistency_scores`, `calibrate_label_consistency`,
`apply_label_consistency_gate`, and immutable result records. The inputs to
scoring are teacher probability rows, supplied replay labels, and explicit
`teacher.class_ids`; for each row, the score is
`1 - P_teacher(supplied_label)`. The supplied label is mapped to its
probability *column* through `class_ids`, never used as an array index.
For a teacher with columns `[10, 0, 5]`, probabilities `[0.1, 0.7, 0.2]`
and supplied label `0` give score `0.3`, not `0.9`.

**Calibration:** the notebook uses the new clean Task 1 teacher on *clean
Task 1 validation only*, computes these scores, and fixes their predeclared
95th percentile using linear quantile interpolation. This threshold is
frozen before attacked conditions and not chosen from their test scores.
Candidates with score `> threshold` are quarantined; equality stays. For
example, threshold `0.25` retains score `0.25` but rejects `0.875`.

**Information boundary:** the gate receives only replay features, supplied
labels, teacher probabilities/class mapping, and the calibrated threshold.
It has no argument for original clean labels or simulator changed indices;
it cannot restore labels. It returns copied retained arrays and masks.
If every row is quarantined, its model-replay adapter returns `None` so the
existing `fit` runs explicitly without replay. The teacher is a frozen
five-class model; using it to filter genuine Task 2 classes 5–7 would
wrongly penalize novelty, so the gate is applied only to old replay rows.
The teacher saw the clean Task 1 distribution, which is a narrow assumption,
not protection against backdoors or arbitrary new-class poisoning.

### Step 3: keep simulator truth in a separate audit

**New code:** `src/evaluation/replay_gate.py` exports
`replay_gate_metrics`. It takes the already decided retained mask plus
simulator-only original labels and changed positions. It verifies that
changed indices correspond exactly to altered labels, then reports poison
rejection, clean false rejection, retained poison fraction, and retained
replay counts per *supplied* class. Clean controls have no poison denominator;
their poison rejection/fraction are `None`, not an invented zero.
For example, if an 80-row targeted poison loses 10 poisoned rows at the
gate, poison rejection is `10/80 = 0.125`; if it also loses 5 of 420 clean
rows, clean false rejection is `5/420`. These audit records must never feed
the gate. This separation makes accidental oracle filtering visible.

### Step 4: prepare six controlled Kaggle conditions

**New notebook:** `notebooks/kaggle_review2_replay_poison_mitigation.ipynb`
pins published gate/audit source
`c636331fc9cdf2361357fd901d9c93411e3f59ba`. It regenerates a fresh
clean Task 1 checkpoint because the attachment has none, reusing the existing
2,000-per-class seeded sampler, fixed split, global-ID task builder,
Task-1-training-only preprocessing, and 100-per-old-class replay selector.
It keeps raw dataset row IDs and the input checksum in its manifest. Task 1
is fitted on IDs 0–4 only. The model and teacher reload are checked before
the gate threshold is calibrated.

The threat model changes **only replay labels** after clean Task 1. The
six conditions are clean replay ± gate, random 20% replay-label flipping
± gate, and targeted 20% old attacks→Benign ± gate. The existing
`apply_label_flip` builds each attack once; its filtered/unfiltered pair uses
the identical supplied labels. Random rate uses all 500 replay rows and
changes 100; targeted rate uses 400 eligible IDs 1–4 and changes 80 to ID
0, only 16% of the full buffer. Both use seed 42, old-class destinations
only, no feature changes. Task 2's 4,200 new-class training rows and the
clean seen-class validation pool are unchanged.

Every Task 2 condition independently reloads the *new* Task 1 checkpoint,
expands global class IDs to 0–7, asserts the complete initial-weight hash
matches all other conditions, resets the training RNG, and fits with the
same architecture/hyperparameters. Different retained replay sizes may
change optimizer steps; the notebook records rows, epochs, steps, and time.
Clean validation loss selects checkpoints, with the test set held out.
After selection it reports old/new/combined classification, per-class
recall/F1, signed forgetting, Benign FPR, old attack-to-Benign errors,
gate audit counts, and checkpoint reload verification. It compares filtered
attacks both to their identical unfiltered attack and to the clean-filtered
control, without assuming improvement.

The timestamped Kaggle folder includes `manifest.json`,
`comparison.json`, `calibration.json`, `row_indices.npz`, frozen
`preprocessing.json`, the clean replay buffer, Task 1 history/metrics and
checkpoint, six separate condition folders with attack/config/gate
audit/history/metrics/checkpoint and per-row audit arrays, plus
`training_curves.png`. The original clean labels and changed masks are
written for post-run auditing only. No replay-poisoning or gate performance
has been observed yet.

### Step 5: verification, limits, and next work

`python3 -m unittest tests.test_label_consistency -v` passed five synthetic
tests for noncontiguous probability-to-class mapping, clean-quantile
calibration, equality/empty gate decisions, non-mutation, invalid inputs,
and random/targeted budget + audit arithmetic. The full local
`python3 -m unittest discover -s tests -p 'test_*.py' -q` ran 32 tests:
31 passed, one opt-in real-data test skipped. `compileall`, notebook
JSON/code syntax and Markdown-stage/empty-output validation, and
`git diff --check` passed. The notebook has 21 cells, 10 code, no saved
results. Local PyTorch/CUDA and the private Kaggle input remain unavailable,
so Task 1 regeneration and all six Task 2 conditions still require Kaggle.
Next phases after inspecting actual execution: stronger replay balancing,
timing backdoors, held-out novelty evaluation, and full-data streaming.

### How to explain this in Review 2

Our first continual run showed that replay helped old-class retention, but
Benign errors remained too high. We now test a narrower attacker: someone
changes labels on saved old examples before we rehearse them. A clean Task 1
teacher asks how well each supplied label agrees with its probability for
that class. We fix a cutoff from clean validation, quarantine suspicious
replay rows, and compare paired clean and attacked updates. We separately
measure how many truly poisoned and clean rows were removed; the filter
does not get that hidden answer key. We will not call it effective until
Kaggle produces and we inspect the six-condition results.

Likely faculty questions:

- **Is the threshold chosen to improve attacked test results?** No. It is
  fixed at the clean Task 1 validation score's 95th percentile first.
- **Does filtering know which labels were flipped?** No. Original labels
  and changed-index masks exist only in the later audit.
- **Why not filter new Task 2 classes?** The old teacher has no output for
  IDs 5–7, so that would conflate novelty with poisoning.
- **Are random and targeted 20% equal budgets?** No: 100/500 versus
  80/400 eligible, or 16% of all replay for targeted.
- **Does this prove backdoor resistance?** No. It examines old replay-label
  inconsistency under one teacher, one subset, and one seed.

---

## 2026-10-03: observed mitigation, optional balanced exposure, and held-out novelty protocol

### Step 1: inspect the mitigation execution before choosing the next test

**Observed, not implemented in this step:** the untracked Kaggle
`data/mitigation/manifest.json`, `comparison.json`, `calibration.json`,
executed `mitigation.ipynb`, and training-curves plot were inspected. The
executed notebook had no error outputs. Its source was
`c636331fc9cdf2361357fd901d9c93411e3f59ba`; the combined-collection
checksum remained `666af53c788c79312b421c607289cb555c2335ae51ba39b2700a357595fa8dba`.
The runtime was Tesla T4, Python 3.13.15, PyTorch 2.11.0+cu128, NumPy
2.1.3, and PyArrow 23.0.1. All six expanded initial-weight hashes matched,
and six checkpoint reload checks passed.

Combined test accuracy (unfiltered/filtered) was clean
`0.7804166667`/`0.81375`, random20 `0.81625`/`0.7858333333`, and
targeted20 `0.81625`/`0.8279166667`. The filter rejected all 100/100
random and 80/80 targeted poisoned replay labels; it also rejected about
4–5% of clean candidates. Benign FPR remained `0.90`–`0.9533333333`.
Optimizer steps were clean 444/444, random20 370/648, and targeted20
444/504 (unfiltered/filtered); retained replay size and early stopping
changed training work. Rejection success is **not** the same as downstream
classification improvement: random20 filtered accuracy was lower than
random20 unfiltered, while the clean and targeted pairs improved. These are
single-seed balanced-subset observations, not a general mitigation claim.
The severe Benign errors motivated a clean exposure comparison before adding
more defense complexity.

### Step 2: implement an optional training-draw change, not a larger memory

**New code:** `src/training/sampling.py` exposes
`inverse_frequency_weights(supplied_labels)`. It returns copied per-row
weights, class counts, and expected class probabilities without touching
features or input labels. For combined Task 2 + replay training, each old
class has 100 supplied rows and each new class has 1,400; weight each old
row `1/100`, each new row `1/1400`. Each class's total weight is 1, so its
*expected* draw probability is `1/8`, not the ordinary shuffled old-class
share `100/4700`. This is expectation, not exact per-epoch balance.

`src/models/tabular_transformer.py` adds `ModelConfig.training_sampler`
(`'shuffled'` by default or `'class_balanced'`) and optional `sampler_seed`.
`TabularTransformerClassifier.fit()` still concatenates the same supplied
Task 2 rows and 500-row `(X_replay, y_replay)` buffer. In balanced mode it
passes those weights to PyTorch `WeightedRandomSampler` with replacement and
`num_samples=4700` (the combined row count), omitting DataLoader shuffle;
ordinary mode keeps its seeded shuffle. One seeded sampler generator lives
through all epochs, so its state advances rather than resetting every epoch.
Repeated *draws* do not create extra stored replay rows. The history records
mode/seed, expected class probabilities, actual class counts per epoch and in
total, generator-state hashes, distinct replay row positions drawn, total
draws, and optimizer steps. These are exposure measurements, not final NIDS
metrics. CPU remains a supported device. The implementation follows the
[PyTorch sampler/DataLoader contract](https://docs.pytorch.org/docs/stable/data.html).

### Step 3: add evaluation functions with explicit information boundaries

**New code:** `src/novelty/calibration.py` computes a fifth-percentile
threshold from maximum probabilities on **known Task 1 validation** only;
it rejects empty/invalid calibration input. `src/evaluation/novelty.py`
summarizes unknown precision/recall/F1, known false-rejection rate, and
recall by *external* held-out class ID, validating that IDs 5–7 correspond
exactly to the true unknown rows. Undefined denominators are `None`, not
invented zero rates. For example, a threshold `0.60` flags confidence
`0.59` but treats `0.60` as known because `detect_unknown()` uses strict
`<`. The maximum probability is the Task 1 model's strongest available
class belief; a low maximum is a useful simple uncertainty signal, but a
confidently wrong unseen attack can still evade it. Probability columns map
through `model.class_ids` (0–4 here), not arbitrary global IDs or row
positions. Validation is needed because malformed probabilities could
otherwise make thresholds or metrics look meaningful; functions do not
mutate supplied arrays.

### Step 4: prepare one auditable Kaggle protocol

**New notebook:** `notebooks/kaggle_review2_replay_balance_novelty.ipynb`
pins published source `4e46f7f8ded77612210237e8dc9f6ea8e83730da`.
It requires Kaggle Internet/GPU and the attached reviewed Parquet checksum,
runs synthetic API tests, samples 2,000 per broad class with seed 42, keeps
the fixed 70/15/15 stratified split and original row IDs, fits preprocessing
only on Task 1 training rows, and selects 100 unique replay rows per old
class. The combined flow collection still has unverified per-row provenance.

One fresh five-output Task 1 model trains only on global IDs 0–4. **Part B
occurs before Task 2 training/head expansion:** it calibrates the predeclared
fifth percentile of Task 1 validation confidence, freezes the threshold,
then compares known Task 1 test and genuinely unseen Task 2 test IDs 5–7.
Future-class validation/test rows never fit preprocessing, train Task 1, or
tune that threshold. The notebook will report binary unknown metrics, known
false rejection, and each unknown class's recall; it does not promise success.

**Part A** independently reloads Task 1's checkpoint for the unchanged
shuffled and new balanced Task 2 arms. Both expand to IDs 0–7 with the same
seed, assert identical initial weights, and share architecture, optimizer,
patience, clean seen-class validation, and *identical* 500-row replay memory.
Each arm's best checkpoint follows clean validation loss. The preferred
training strategy is predeclared as higher seen-class validation macro-F1,
with lower validation Benign FPR as an exact-tie breaker. Test and novelty
scores do not choose it. After this choice, both arms report old/new/combined
classification, fixed-five-class forgetting (including negative improvements),
per-class recall/F1, Benign FPR, draw counts, optimizer steps, and checkpoint
reload identity. The timestamped Kaggle output saves source/environment,
checksum, split/replay IDs, preprocessing state, histories, selection record,
novelty predictions/metrics, test metrics, plots, and checkpoints.

### Step 5: verification, limits, and continuation

`python3 -m unittest discover -s tests -p 'test_*.py' -q` ran 39 tests:
36 passed, three skipped (two PyTorch-gated sampler tests and the opt-in
real-data model test). Focused synthetic tests cover weights, default
compatibility, deterministic draws, advancing RNG, strict threshold equality,
external IDs, non-mutation, and novelty metric edge cases. Source compilation
passed. Notebook JSON and all code-cell syntax parse, with no fabricated
execution outputs. Local PyTorch/CUDA and the private Kaggle input are not
available, so the torch-dependent sampler checks and this notebook's training
and held-out novelty evaluation must run on Kaggle. No balanced-exposure or
held-out novelty result exists yet. The following phase, **after** reviewing
this run, is streaming preparation and throughput benchmarking.

### How to explain this in Review 2

Our replay buffer stores only 500 old examples. In ordinary training, they
are a small minority among 4,700 rows each epoch; the new option changes
which rows are *drawn* more often without storing more data. We compare that
single change from the same Task 1 weights and choose the preferred method
using validation, not test data. Separately, before the model learns new
classes, we ask whether low confidence identifies those genuinely unseen
classes. The threshold comes only from known validation examples. We will
show the actual results after Kaggle runs; neither good poison rejection nor
balanced sampling alone guarantees better intrusion detection.

Likely faculty questions:

- **Are balanced draws the same as a balanced memory?** No. Memory remains
  100 unique rows per old class; replacement draws can repeat rows.
- **Are the two arms equal compute?** They have the same rows *drawn per
  epoch* and batch size, but early stopping may yield different epochs and
  optimizer steps; those are reported.
- **How is the threshold chosen?** The fifth percentile of maximum
  probability on clean Task 1 validation only, before unknown test inspection.
- **What is truly unseen?** Task 2 IDs 5–7 are absent from Task 1 training
  and its five-output head; they are transformed with frozen Task-1-only
  preprocessing and evaluated only afterward.
- **Can this prove novelty performance broadly?** No. It tests one held-out
  class split, one seed, and one balanced subset of a collection with
  unverified source provenance. A low confidence score need not reliably
  distinguish every new attack.

---

## 2026-10-03: repair Kaggle notebook test discovery

**Observed failure:** the first Kaggle attempt stopped in the environment/
synthetic-test cell of `kaggle_review2_replay_balance_novelty.ipynb` with
`ModuleNotFoundError` for `tests.test_training_sampling` and
`tests.test_novelty_evaluation`. This happened before data scanning,
Task 1 training, replay comparison, or held-out novelty evaluation. Both
test files are in pinned source
`4e46f7f8ded77612210237e8dc9f6ea8e83730da`, but `tests/` lacks
`__init__.py`, so a dotted package import was the wrong invocation. No
experiment metric can be taken from this stopped run.

**Change and reason:** only the notebook's stage-2 code cell changed. It
now runs two separate subprocesses equivalent to
`python -m unittest discover -s tests -p test_training_sampling.py -v`
and
`python -m unittest discover -s tests -p test_novelty_evaluation.py -v`,
each from the cloned `REPO_DIR` with `check=True`. The code displays each
suite's output and parses its `Ran N tests` summary, raising an error if a
suite discovers zero tests. This keeps a false green test stage from letting
the NIDS experiment proceed. The input is the two exact test filenames;
the output is a visible, nonzero passing test count or an immediate stop.
No source code or model/data/novelty settings changed; the pinned SHA stays
the same. For a concrete example, the local sampler discovery reports
`Ran 4 tests`, so it passes the count guard even though two torch-dependent
tests skip locally; an empty pattern reporting `Ran 0 tests` would stop.

**Verification:** fetched origin and confirmed local/remote branch HEAD
`0315bc041acec0a69629fda0764d43836dda576c` before editing. The two
exact-pattern local discovery commands found 4 sampler tests (2 pass, 2
PyTorch-unavailable skips) and 3 novelty tests (3 pass), with zero failures.
`python3 -m unittest discover -s tests -p 'test_*.py' -q` ran 39 tests:
36 passed, 3 skipped, 0 failed. JSON parsing and `ast.parse` passed for
all 11 notebook code cells; each still has Markdown immediately before it,
and all execution counts/outputs remain empty. The notebook's source SHA
assertion remains unchanged; `git diff --check` passed. These local checks
verify invocation and syntax, **not** Kaggle T4 execution. The corrected
notebook must be rerun with Internet, GPU, and the attached reviewed Parquet.
Then inspect the produced manifest, exposure, novelty, and reload checks
before the following streaming/throughput phase.

**Review-2 explanation:** the run stopped because our notebook asked Python
to import test files as a package that the repository does not define. We
changed the notebook to discover those files by exact filename, require
nonzero test counts, and still halt on any failure. We have no new model or
novelty result from the stopped attempt.

---

## 2026-10-03: observed replay/novelty tradeoff and large-data preparation

### Step 1: distinguish the completed Kaggle run from this new code

**Observed attachment, not newly trained locally:** `data/replay/replay.ipynb`,
its manifest, `test_summary.json`, `strategy_selection.json`, `novelty.json`,
and plot. The notebook recorded no error outputs, seven targeted tests
passed (four sampler, three novelty), and Task 1 plus both Task 2 checkpoint
reload predictions matched. Source was
`4e46f7f8ded77612210237e8dc9f6ea8e83730da`, input checksum
`666af53c788c79312b421c607289cb555c2335ae51ba39b2700a357595fa8dba`,
runtime Tesla T4/Python 3.13.15/PyTorch 2.11.0+cu128.

The predeclared **validation** macro-F1 selected class-balanced sampling:
`0.8529213667` versus `0.7541214312` shuffled. On held-out test rows,
combined accuracy was `0.8508333333` versus `0.7804166667`; old-class
accuracy `0.8353333333` versus `0.6826666667`; new-class accuracy fell to
`0.8766666667` from `0.9433333333` (balanced versus shuffled). Balanced
Benign FPR was still `0.6166666667`, although shuffled was `0.9366666667`.
The arms took 370 and 444 optimizer steps (10 versus 12 epochs), so they
are not equal-compute. The exposure plot shows balanced draws near 1/8 per
class, while shuffled reflects 100 old versus 1,400 new rows per class.
One seed, one 2,000-per-class subset, and unknown per-row source provenance
limit generalization.

For actual held-out novelty, the Task 1 five-output model used only known
validation confidence to freeze a fifth-percentile threshold of
`0.9247980118`. It flagged 170/900 unseen Task 2 test rows: unknown recall
`0.1888888889`, precision `0.7142857143`; it falsely rejected 68/1500 known
Task 1 test rows (`0.0453333333`). Per-class unknown recall was 0.11
Infiltration, 0.1033 Webattack, and 0.3533 Portscan. The plot shows high
confidence for many unknowns. This is real evidence that maximum-confidence
thresholding is weak for this split, not evidence of a reliable novelty
detector.

### Step 2: persist the full row-level split without full feature loading

**New code:** `src/data/large_data.py` provides
`prepare_full_partitions(parquet_path, output_dir, seed=42)` and
`verify_full_partitions(...)`. It scans only `ClassLabel` in existing bounded
Parquet batches, maps the documented eight names to global IDs 0–7, and
assigns every original row once to a seeded 70/15/15 class-stratified
partition. Persisted `labels.npy`, `splits.npy`, `row_ids.npy`, and explicit
`train_indices.npy`, `validation_indices.npy`, `test_indices.npy` retain the
mapping. The verifier checks sequential original IDs, valid class/split
codes, complete coverage, per-class counts, and each index file against
the split codes. A three-row example with row IDs `[0,1,2]` and split codes
`[0,1,2]` means one train, one validation, one test row; no ID is duplicated.
The actual 9,167,581-row counts are **not yet generated**. Row-level
stratification cannot guarantee independence among duplicate flows or flows
from one capture session because the collection has no session provenance.

### Step 3: write bounded features and fit two separate states

`prepare_raw_feature_store()` reads only the 54 retained numeric columns
plus `ClassLabel` for an alignment check; `Label` and `ClassLabel` never
enter `X`. Each bounded batch writes row-aligned float32 values to one
`raw_features.npy` memory map. Negatives and non-finite values become NaN,
as in the existing small-array pipeline. It does not allocate the full
float64 source matrix or concatenate whole partitions. A feature row's
array position remains its original Parquet row ID; for example, stored
row 42 pairs with label and split entries 42.

`fit_disk_preprocessor(root, scope='static'|'continual')` makes **separate**
preprocessing records. Static fit can see all rows with train split code;
continual fit can see only train rows whose global IDs are 0–4. A seeded
without-replacement reservoir of at most 200,000 authorized rows estimates
each median. This is deliberately **approximate** (unlike the original
small-array exact median); its seed, size, scope, and method are recorded.
After imputation, Welford-style streaming batch moments compute population
mean/std over every authorized training row. Validation/test and future
Task 2 rows never fit the continual state. In a tiny example where old
train feature values are 1 and new train values are 101, continual mean
stays 1 while static mean changes; a validation value of 10,000 changes
neither fit. Each state freezes before future transformation.

Completed split, feature-store, and state records can be reused on a restart;
an interrupted unmarked stage rebuilds its partial file. The Kaggle
notebook verifies the input checksum first and checks disk capacity. This
does not make the input provenance known or repair duplicate-flow leakage.

### Step 4: offer bounded-batch training and measure only throughput

**New code:** `src/training/disk_backed.py` exposes
`DiskBackedFlowDataset(root, preprocessor, partition, class_ids, row_ids=None)`
and `fit_disk_backed(model, train, validation=None, ...)`. The view holds
integer row positions and a read-only memory map; each batch loads only its
requested raw rows, transforms with the frozen state, and returns float32
features, int64 global labels, and original IDs. It rejects class or split
violations at view construction. Shuffled order uses a seed and epoch;
optional balanced order samples from per-class integer buckets with
replacement, so no float64 weight per full-data row is allocated. The
trainer uses the existing Transformer network/config/class IDs, AdamW,
cross-entropy, checkpoint save/load format, and clean validation-loss
best-state/early-stopping logic. Existing `fit(X, y, ...)` remains unchanged.
The new disk fit can run without validation for a **training-only**
throughput benchmark; that case selects no checkpoint and makes no claim
about classification quality.

**New notebook:** `notebooks/kaggle_review2_large_data_prepare.ipynb` pins
published source `5c0d519596bc6c4b99948126443635c2f7eb15f1`. It
checks the exact input hash, 9,167,581 row count, 54-column schema, T4,
package versions, and free working space; runs synthetic checks; prepares
all original-row partitions and both fit states; then chooses up to 100,000
**training-only** rows uniformly with seed 42, retaining their natural class
distribution. Fresh hidden-64/two-layer models receive one training epoch
at predeclared batch sizes 128 and 256 with mixed precision off. CUDA is
synchronized around timings. The notebook records actual draws, optimizer
steps, measured rows/second, process peak RAM, GPU peak allocated/reserved
memory, preparation/storage cost, and a labeled full-epoch planning estimate
with 1.5× headroom. Its provisional recommended batch size follows measured
throughput only; no test metric is opened or tuned. The output folder has
manifests, arrays, preprocessing JSON, benchmark JSON, and configuration.
Preserve it as a Kaggle output/private dataset for the next session; never
commit it.

### Step 5: verify honestly and defer real large-data training

`python3 -m unittest discover -s tests -p 'test_large_data.py' -v` ran four
synthetic tests: three passed and one torch-dependent CPU trainer test
skipped because local PyTorch is absent. They cover deterministic complete
split/restart, targets excluded, row/batch alignment, static versus
Task-1-only fit, validation/test leakage rejection, stable balanced draws,
and unchanged small-array preprocessing. Full
`python3 -m unittest discover -s tests -p 'test_*.py' -q` ran 43 tests:
39 passed, four skipped (three PyTorch/real-data gated plus the new trainer
test). `python3 -m compileall -q src tests`, notebook JSON parsing and
code-cell `ast.parse`, empty-output/Markdown checks, and `git diff --check`
passed. The full Parquet preparation, T4 benchmark, exact Kaggle disk cost,
and GPU trainer are **not executed yet**. Published source was pinned so
Kaggle can run its torch-dependent synthetic test. After reviewing actual
preparation throughput and capacity, the next phase is real full-data
training—not another attack family or the full experiment suite here.

### How to explain this in Review 2

The balanced subset taught us that replay exposure helped old classes, but
Benign errors remained high and simple confidence missed most truly unseen
attacks. For the full collection, we now separate *preparation* from
*training*: each original row receives one reproducible split, feature
batches stream to disk, and separate train-only statistics are frozen for
static and continual protocols. We will first measure T4 throughput on
100,000 training rows and use that only to plan resources. We have not
trained or evaluated the full 9.17-million-row model yet.

Likely faculty questions:

- **Can row-level splits leak related traffic?** Yes. Without flow/session
  provenance or duplicate groups, class stratification alone cannot rule it
  out; full-data scores will need that caveat.
- **Are the medians exact?** No. They are seeded approximate medians from up
  to 200,000 authorized training rows; means/stds stream all authorized
  training rows after imputation.
- **Why two preprocessing states?** Static training may use all eight-class
  training rows; Task 1 must not learn feature statistics from future Task 2.
- **Is the full-epoch time a result?** No. Only subset throughput is measured;
  the full-epoch figure is a 1.5×-headroom planning estimate.

---

## 2026-10-03: completed preparation evidence and durable full clean run

### Step 1: verify the attached preparation record before scheduling training

**Observed Kaggle execution, not new local training:** inspected
`data/preparation/preparation.ipynb` and all supplied preparation JSON
artifacts. Eight notebook code cells executed with no error outputs. Source
revision was `5c0d519596bc6c4b99948126443635c2f7eb15f1`, input SHA-256
`666af53c788c79312b421c607289cb555c2335ae51ba39b2700a357595fa8dba`,
and runtime Tesla T4/Python 3.13.15/PyTorch 2.11.0+cu128. The manifest and
split record agree on **6,417,308 train, 1,375,139 validation, 1,375,134
test** original rows, totaling 9,167,581. The stored feature schema has 54
float32 columns and excludes both `Label` and `ClassLabel`. Static state
used the full 6,417,308 training rows for streaming moments; continual
state used 6,347,232 Task-1-only training rows. Each median approximation
used a seed-42 reservoir of 200,000 authorized rows. The reported persisted
output was about 2.146 GB. The actual large `.npy` arrays were **not**
attached locally, so their contents have not been independently audited
here; the next notebook validates them before use.

Batch 256's **measured** one-epoch benchmark on 100,000 training rows was
4.5053 seconds, 391 optimizer steps, and 22,196.07 rows/s; batch 128 was
12,079.26 rows/s. The reported 433.68 seconds for one full training epoch
is a **projection** using 1.5× headroom and 6,417,308 rows. It excludes
the 1,375,139-row validation pass, checkpoint writes, and session startup;
it must not be presented as measured full training time. No full-data model
accuracy or test score exists yet. The extreme natural class distribution
(for example, 5,030,332 Benign versus 1,579 Portscan train rows) makes
accuracy alone misleading.

### Step 2: make epoch-boundary resume durable and auditable

**New code:** `src/training/full_run.py` adds
`train_full_disk_backed(model, train, validation, output_dir,
data_identity, resume=False, ...)`. It consumes the existing
`DiskBackedFlowDataset` views; every training batch loads only its rows and
uses the existing eight-output Transformer, AdamW, cross-entropy, and global
class-ID mapping. Ordinary seeded shuffled order is used for this static
baseline. Full validation is streamed in batches after every epoch; only
strictly lower full validation loss updates the best state. The test view
does not exist in this stage. Each epoch records actual training and
validation rows, losses, validation accuracy, optimizer steps, synchronized
train/validation seconds, and a measured-rate remaining-time estimate.

`latest.pt` is the authoritative atomic epoch snapshot. It stores current
model weights, optimizer buffers, scaler state, completed epoch, history,
best-validation weights/loss/epoch, stale/patience counter, and Python,
NumPy, Torch CPU, and Torch CUDA RNG states. `best.pt` is loadable through
the existing `TabularTransformerClassifier.load()` API. A JSON history and
training manifest are written after each completed epoch. On resume, the
caller supplies the same model configuration, class IDs, prepared-data
identity, and PyTorch/device runtime; mismatches stop instead of silently
starting a new optimizer. The trainer restores optimizer/scaler and all RNG
states, resumes at the next deterministic epoch, and repairs `best.pt` from
authoritative latest if a crash occurred between saves. An interrupted
partial epoch repeats from the last completed boundary. It does **not**
promise mid-epoch continuation or bitwise-identical results across changed
GPU/PyTorch runtimes.

Concrete example: if epoch 3 finishes and `latest.pt` records 3 plus AdamW
moments and a patience count of 2, then a session interruption during epoch
4 leads the next session to reload that exact epoch-3 state and repeat epoch
4's seeded order. It does not reset weights, AdamW moments, dropout RNG, or
patience to zero. A new run with a different preprocessing hash is rejected.
The checkpoint contents follow [PyTorch's general-resume guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html),
with explicit CPU/CUDA RNG snapshots added for this protocol.

### Step 3: evaluate a selected checkpoint with bounded memory

**New code:** `src/evaluation/streaming.py` provides
`ConfusionAccumulator(class_ids)` and `evaluate_disk_backed(model, test,
batch_size, ...)`. Each test batch produces external class IDs by mapping
the network output column through `model.class_ids`; only an 8×8 count
matrix survives. The final counts yield accuracy, macro-F1, balanced
accuracy (mean of the eight recalls), each class's precision/recall/F1/
support, and Benign FPR. For example, if two Benign rows yield one Benign
and one attack prediction, Benign recall is 1/2 and FPR is 1/2; accumulating
those rows in separate batches yields the same matrix. This avoids a
1.375-million-row prediction vector in memory. The previous small-array
evaluator and `fit()` API remain unchanged.

### Step 4: prepare the Kaggle full clean notebook, not a result

**New notebook:** `notebooks/kaggle_review2_full_clean_training.ipynb` pins
published API source `91a26a1c13a563a7628e217f357dc27f73874784`.
`PREPARED_DIR` selects the existing `/kaggle/working` folder or an attached
private `/kaggle/input` folder. Before training, it compares preparation
source/input identity, split counts/coverage, row IDs, array shapes/dtypes,
54-feature order, target exclusions, and static/continual fit scopes. It
opens prepared arrays read-only and hashes the raw store, labels, splits,
static preprocessing state, and split manifest; these hashes become the
resume identity. It never repeats preparation or swaps in the old balanced
subset. A separate `/kaggle/working/full_clean_...` folder holds outputs.

The initial configuration is fixed: fresh eight-class hidden64, four heads,
two layers, MLP128, batch256, lr0.001, seed42, mixed precision off,
`training_sampler='shuffled'`, max20 epochs, and clean full-validation-loss
patience5. The notebook runs focused synthetic tests first. It can copy
small prior run checkpoints from a saved Kaggle input into a new writable
run folder using `RESUME_FROM`; the 2.1-GB prepared arrays stay read-only.
It trains all 6,417,308 natural-distribution training rows, selects the
best checkpoint **only from full validation loss**, verifies reload against
the authoritative best-state snapshot, and only then evaluates the full
1,375,134-row test split **once** with bounded inference. A matching saved
test record is reused on notebook rerun. It saves history, manifests,
config/environment/data hashes, best/latest checkpoints, reload audit,
curves, confusion matrix, and test metrics. These are planned outputs, not
observed results. Full-data continual/replay and poisoning comparisons are
subsequent separate runs.

### Step 5: verification, limitations, and Review-2 explanation

Local `python3 -m unittest discover -s tests -p
'test_streaming_evaluation.py' -v`: 2 passed. The resume suite found two
tests: one pure metadata-compatibility test passed; one synthetic PyTorch
CPU uninterrupted-versus-resumed weight/loss equivalence test skipped
because PyTorch is unavailable locally. Full
`python3 -m unittest discover -s tests -p 'test_*.py' -q`: 47 tests,
42 passed, 5 skipped, zero failed. `python3 -m compileall -q src tests`,
notebook JSON parse and code-cell `ast.parse`, Markdown-before-code and
empty-output checks, and `git diff --check` passed. The Kaggle notebook
reruns the PyTorch-dependent tests before accessing prepared data. No full
training, full validation, or full test inference occurred locally or in
the supplied preparation run. The static medians are still approximate,
row-level duplicate/session leakage remains possible, and runtime changes
can prevent exact resumption.

**How to explain this in Review 2:** We have prepared all rows once and
measured only subset throughput. The new notebook will train the clean
eight-class model on every natural training row, check the entire
validation set each epoch, and protect multi-session progress with complete
epoch checkpoints. Only after validation fixes the best model will it open
the untouched test partition. We will report all class recalls and Benign
false alarms, not just overall accuracy, because Benign dominates this
collection. We cannot yet quote a full-data accuracy: that run has not
executed.

Likely faculty questions:

- **Does resume restore only the model?** No: it also restores AdamW,
  scaler, RNG, history, best state, and patience; mismatched data/config
  stop the run.
- **Can a crash lose work?** At most the currently incomplete epoch; the
  latest completed epoch is saved atomically.
- **Why use validation loss rather than test metrics?** Test is reserved for
  one final unbiased assessment after model selection.
- **Why report balanced accuracy?** Its mean per-class recall does not let
  the large Benign class dominate the summary the way ordinary accuracy can.

---

## Full clean result review and full-data continual Task 1 foundation (2026-10-03)

### Step 1: inspect what actually ran

**Existing code/run, not new training this task.** The attached
`data/full-train-v2/results/full_clean_20261003T045146_553311Z/` contains
the eight-class static run's supplied manifests, per-epoch history,
test confusion counts/metrics, environment, reload report, and two plots.
The separate `data/full-train.ipynb` has three `NameError` outputs, so we
do not treat that notebook as evidence of the completed v2 run; the
completed v2 artifacts support the observations below.
The run used published source `91a26a1c13a563a7628e217f357dc27f73874784`
and the previously prepared 9,167,581-row combined flow collection.
It completed 19 epochs (five-epoch early stopping), with epoch 14 selected
by **validation loss**. The training manifest sums to `5904.202296291`
training seconds plus `353.498714439` validation seconds, or
`104.2950168455` minutes; this excludes separate test inference. The
full 1,375,134-row test yielded accuracy `0.9828620338090688`, macro-F1
`0.757755762638612`, balanced accuracy `0.7291289708762002`, and Benign
FPR `0.00445669840657261` (4,804 false alarms among 1,077,928 Benign
rows). Infiltration recall was `0.059108799550182736`: 13,328 of 14,228
were called Benign. Webattack recall was `0.12694877505567928`: 384 of
449 were called Benign. Thus high accuracy is mainly a majority-class
summary, not evidence of uniformly strong attack detection. The raw-count
plot visually hides minority rows because Benign is so numerous. The
supplied reload JSON states best-state and bounded prediction agreement;
we did not independently load `.pt` weights locally without PyTorch.
This is one split, one seed, a combined collection of unverified per-row
source provenance, and a static classifier—not production or continual
performance. These test results are a reference, not a tuning set for
Task 1 architecture, stopping, or novelty threshold.

### Step 2: derive clearer plots from saved counts

**New code:** `src/evaluation/plots.py` exposes
`confusion_plot_data(record)` and `save_result_plots(metrics_path,
output_dir, prefix='test')`. The input is the already saved metrics JSON,
including global `class_ids`, true-row/predicted-column integer confusion
counts, and class names. It returns/writes a separate raw-count PNG,
an annotated row-normalized heatmap, and a per-class recall/F1 chart.
For a true Infiltration row with 5 predictions of Infiltration and 95 of
Benign, normalization displays 5% on the diagonal and 95% under Benign,
regardless of how many Benign rows the dataset has. Recall is diagonal
count divided by true-class support; F1 also accounts for false predictions
*into* that class. A zero-support row is zero rather than divided by zero.
The utility validates nonnegative integer counts and external class IDs,
does not mutate its input, does not overwrite plots, and never opens a
checkpoint or test dataset. The original `confusion_matrix.png` remains
untouched. Use, for example:

```python
from src.evaluation.plots import save_result_plots
save_result_plots("test_metrics.json", "derived_plots", prefix="static_clean")
```

**Verified:** `python3 -m unittest discover -s tests -p
'test_result_plots.py' -v` passed three synthetic tests (external ID mapping,
zero-support/row normalization, non-mutation, files and no overwrite,
invalid counts). The utility also generated all three PNGs from the
attached static `test_metrics.json` in a temporary local folder. This
reused saved counts only; it was not a second test inference. It does not
change metrics or correct the model's minority weakness.

### Step 3: select replay positions safely, without training Task 2

**New code:** `src/continual_learning/disk_replay.py` adds
`select_disk_replay_rows(dataset, class_ids, per_class=100, seed=42)`.
It requires a disk-backed **training** view, checks each candidate's split
code, and samples unique original positions without replacement. Output
is just an integer position vector; `DiskBackedFlowDataset.batch()` then
reads and transforms only the 500 selected features using the frozen
Task-1-only state. The notebook saves those features, labels, and original
row IDs as `replay_buffer.npz` plus an auditable manifest. For a toy view
with three training rows each of IDs 0 and 1, `per_class=2` returns four
unique training positions—never validation/test positions. No replay rows
enter Task 1 model fitting; they are a future Task 2 resource.

**Verified:** `python3 -m unittest discover -s tests -p
'test_disk_replay.py' -v` passed two synthetic tests: deterministic/balanced
non-mutating selection and rejection of future/validation or insufficient
rows. The full prepared arrays were not available locally to sample now.

### Step 4: prepare the five-output full Task 1 notebook

**New, unexecuted notebook:**
`notebooks/kaggle_review2_full_task1_training.ipynb` pins published source
`e6ebfcf7a9beec4d2b8fd5245e7c62314b9fadf9`. It reads the *existing*
prepared folder in `/kaggle/working` or `/kaggle/input`, validates its
source/input SHA, split coverage and counts, 54-feature order, raw array
shapes/dtypes, and both preprocessing scopes. It hashes raw features,
labels, splits, original IDs, split manifest, and specifically
`preprocessing_continual.json`; the hash enters its strict resume identity.
It constructs the continual `FittedPreprocessor`, not the static one. Its
median was estimated from a recorded 200,000-row seed-42 reservoir of
Task 1 training only, and its streaming mean/std fit from all 6,347,232
Task 1 training rows. Neither Task 2 nor validation/test fit this state.

It creates a **fresh** five-output model for global IDs 0–4 with hidden64,
four heads, two layers, MLP128, batch256, lr0.001, seed42, ordinary shuffle,
mixed precision off, max12 epochs, and patience5. The 12-epoch cap is a
time-budget decision made **before any full Task 1 result was observed**;
all other settings are unchanged. `train_full_disk_backed`
processes all 6,347,232 old-class training rows and all old-class validation
rows in bounded batches, checkpointing latest/best each completed epoch.
No eight-class static weights or future-class validation labels enter Task 1.
For a resumed run, model, AdamW, scaler, RNG, history, best-state, and
patience are restored only when configuration/runtime/data hashes match.
An interrupted partial epoch repeats from the prior completed boundary.

After validation-loss selection, the notebook verifies best reload, saves
training curves and the 500-row clean replay buffer, then calibrates a
maximum-confidence threshold as the fifth percentile of **Task 1
validation** scores. For instance, if known-validation maxima have fifth
percentile 0.62, a later test flow at 0.61 is flagged unknown but one at
0.62 is known. The threshold is frozen before either test group is read.
Bounded inference with the original five-output head evaluates known Task 1
test versus genuinely unseen Task 2 test rows; Task 2 is not trained or
used for calibration. The known pass also accumulates Task 1 confusion
counts once. `novelty_metrics` reports unknown precision/recall/F1,
known false-rejection, and per-unknown-class recall. These are planned
outputs until Kaggle runs. Confidence can still miss new attacks.

The output folder holds config/environment/data identity, a verified copy
of `preprocessing_continual.json`, `latest.pt`, `best.pt`, per-epoch history,
replay positions/features/labels,
reload record, old-class test and novelty JSON, raw/normalized confusion
plots, class recall/F1 chart, and a final manifest. The notebook supports
`RESUME_FROM` for an existing working folder or a saved Kaggle input and
does not silently substitute the earlier balanced subset. Save the entire
folder privately for the next session; never commit the arrays/checkpoints.
Task 2 sequential/replay arms are a **separate next phase** after reviewing
the actual Task 1 run.

**Pre-run time-budget amendment:** The initially prepared notebook capped
Task 1 at 20 epochs, matching the earlier static baseline configuration.
Before Task 1 execution or test inspection, we changed only its cap to 12.
Patience remains 5, so training can still stop earlier when validation loss
fails to improve; if it reaches epoch 12, the best validation-loss checkpoint
among completed epochs is used. Per-epoch `latest.pt`/`best.pt` writes,
strict epoch-boundary resume, subsequent old-class/held-out evaluation, and
replay export are unchanged. Because epoch count is part of the persisted
configuration, a hypothetical 20-epoch Task 1 checkpoint would be rejected
by the 12-epoch notebook rather than silently resumed. The observed
eight-class static baseline and its 19 completed epochs were not touched.
The revised notebook parsed as JSON; all 8 code cells passed Python syntax
checks, with 17 cells total and no execution outputs. The recorded config
was checked for `epochs=12` and `early_stopping_patience=5`, and the resume,
evaluation, replay-export, and published-source references remain present.
`python3 -m unittest discover -s tests -p 'test_*.py' -q` ran 52 tests
(47 passed, 5 existing environment/opt-in skips); `python3 -m compileall
-q src tests` and `git diff --check` passed. This verifies preparation,
not Kaggle execution or a Task 1 metric.

### Step 5: verification and limits

The source synthetic tests and attached-count plotting passed as described
above. `python3 -m unittest discover -s tests -p 'test_*.py' -q` ran 52
tests: 47 passed, 5 skipped, zero failed. `python3 -m compileall -q src
tests` passed. Local notebook JSON parse/`ast.parse` verified 17 cells,
8 code cells, Markdown before stages, empty outputs, and the published
source pin; `git diff --check` passed. Kaggle GPU training,
full-data Task 1 test, and held-out novelty results have **not** yet run.
Local PyTorch-dependent resume-equivalence checks remain skipped without
PyTorch. Row-level splits cannot guarantee duplicate-flow or session
independence, and approximate medians plus one seed limit interpretation.

**How to explain this in Review 2:** The full static model tested well on
the majority of flows, but its Infiltration and Webattack recalls were
low. We used saved confusion counts to show that clearly without changing
the model or rerunning the test. Next we will build a separate five-class
Task 1 model from the same full prepared collection, using a preprocessor
that has seen only old-class training data. We will save its best validated
checkpoint and a clean 500-flow replay memory, then test whether its
confidence distinguishes held-out new classes. Only after reviewing that
foundation will we train Task 2.

Likely faculty questions:

- **Why not reuse the eight-class static checkpoint?** It already learned
  the Task 2 classes, so it would invalidate a class-incremental baseline.
- **Why does Task 1 preprocessing exclude Task 2 training rows?** Otherwise
  even feature statistics would leak future-task information.
- **Does high static accuracy mean minority attacks are solved?** No;
  Infiltration recall was about 5.9% and Webattack about 12.7%.
- **Is the 500-row buffer used in this run?** It is selected and saved from
  old-class training rows only; Task 2 replay has not begun.
- **Is novelty success guaranteed?** No; the fifth-percentile threshold is
  a predeclared confidence baseline, evaluated only after calibration.

---

## Full Task 1 result and bounded targeted Task 2 replay study (2026-10-03)

### Step 1: distinguish observed Task 1 evidence from the unrun Task 2 plan

**Existing run reviewed:** The supplied
`data/full-run-task_01/results/full_task1_20261003T074354_763068Z/`
manifests, history, metrics, novelty JSON, replay manifest/buffer,
environment and plots report a Tesla T4/Python 3.13.15/PyTorch
2.11.0+cu128 run from source
`e6ebfcf7a9beec4d2b8fd5245e7c62314b9fadf9`. Task 1 used global IDs
0–4, 6,347,232 training rows and 1,360,123 validation rows. It completed
12 epochs with epoch 9 selected by validation loss. Training plus validation
totaled `65.50660256425` minutes. On 1,360,119 old-class test rows, the
saved metrics report accuracy `0.9925734439413022`, macro-F1
`0.981162951652861`, balanced accuracy `0.9784715590433037`, and Benign
FPR `0.004171892742372403` (4,497 false alarms out of 1,077,928 Benign).
The saved row-normalized confusion plot and class chart agree with strong
old-class recall, although DoS recall is lower than the other old classes.
These are observed Task 1 results, not Task 2 retention.

The predeclared fifth-percentile maximum-confidence novelty threshold
flagged 691/15,015 held-out Task 2 test rows, unknown recall
`0.04602064602064602`. It also rejected 67,987/1,360,119 known Task 1
test rows. The threshold is a simple baseline; this is **low** unknown
recall, not evidence of robust unseen-attack detection. We did not lower
or tune it from these test results. The separate attached notebook snapshot
has no execution outputs, so observations are attributed to the completed
manifests/metrics instead.

The Task 1 replay `.npz` is present locally. We independently checked its
`(500,54)` finite float32 features, 500 unique int64 original IDs, 100
labels per old class and exact ID agreement with its manifest. We also
hashed the attached `best.pt` file and matched the reported SHA-256. We
could not load its weights without local PyTorch or compare replay IDs and
features to the unavailable full prepared arrays. The new Kaggle notebook
performs those checks before using either input. Thus file-hash and buffer
structure verification are distinct from independent weight/provenance
verification.

### Step 2: construct equal-work Task 2 sampling without a full feature copy

**New code:** `src/training/task2_replay.py` provides `Task2ReplayView`.
Inputs are a disk-backed view of all 70,076 clean Task 2 training rows
(global IDs 5–7), already transformed frozen replay features/labels/IDs,
and a fixed 70,576-draw epoch budget. It validates that replay IDs come
from old-class **training** rows. The view never calls preprocessing on
replay features again; only new rows are read/transformed from the prepared
memmap in bounded batches. For each draw, it picks uniformly among classes
present under the *supplied* labels, then uniformly among that class's
available examples, with replacement. An epoch-specific seed advances the
sequence. For example, if filtered replay loses every class-2 candidate,
classes 0,1,3,4,5,6,7 each have expected draw probability 1/7; the epoch
still draws exactly 70,576 rows rather than doing less work. This does not
increase the 500-row replay memory. Actual per-class draw counts, expected
probability, unique replay examples drawn, and optimizer steps enter each
epoch's history through a backward-compatible optional exposure hook in
`train_full_disk_backed`. Its existing ordinary full-disk and small-array
paths are unchanged.

**Verified synthetic:** `python3 -m unittest discover -s tests -p
'test_task2_replay.py' -v` passed 2 tests for fixed balanced draws,
seeded epoch advancement, no replay double-transform/mutation, empty old
replay and split rejection. PyTorch-based full training remains a Kaggle
check, not a local pass.

### Step 3: freeze the threat model and keep the gate blind to simulator truth

**Existing attack/gate reused, new bounded calibration helper:**
`apply_label_flip` chooses 80 of 400 eligible old attack replay labels
(IDs 1–4) using seed 42 and assigns Benign ID 0. It changes 20% of
eligible sources, 16% of the full 500-row buffer. The two attacked arms
share exactly this corrupted supplied-label array; Task 2 training labels,
replay features, validation, teacher weights and preprocessing remain clean.
The notebook saves changed original IDs and both denominators in an audit.

The frozen five-output Task 1 teacher scores only **clean Task 1
validation** in batches with `label_inconsistency_scores`, mapping each
supplied external ID via `teacher.class_ids`. New
`calibrate_label_consistency_scores` takes the retained one-dimensional
clean scores and fixes their 95th percentile; it is equivalent to the
existing probability-matrix calibration but avoids holding the full
1.36-million-row probability matrix. For a replay candidate supplied as
Benign with teacher probability 0.10 for Benign, inconsistency is 0.90.
`apply_label_consistency_gate` rejects only scores strictly *above* the
frozen threshold. Its inputs are features, supplied labels, teacher
probabilities/class mapping, and threshold—**not** original clean labels,
changed positions, or future Task 2 examples. After decisions,
`replay_gate_metrics` separately uses simulator truth to report poison
rejection, clean false rejection, retained poison fraction, and class
counts. This is an experimental exclusion rule, not a persistent
review/release system or proof against adaptive/backdoor attacks.

**Verified synthetic:** `python3 -m unittest discover -s tests -p
'test_label_consistency.py' -v` passed 6 tests, including agreement between
old and new calibration APIs, non-mutation, threshold/gate behavior,
class-ID mapping and separate audit accounting. No full-data filter result
is claimed until the notebook runs.

### Step 4: train and evaluate three fixed arms from the same Task 1 state

**New, unexecuted notebook:**
`notebooks/kaggle_review2_full_task2_targeted_mitigation.ipynb` pins
published source `857dfbbcca2cfd6b33041ded51fbbae39b7a43f2`
(the balanced-accuracy follow-up to `5546e7a4165eea51ba89e912603b5387e9506703`). Its
explicit `PREPARED_DIR` and `TASK1_RUN_DIR` selectors accept current Kaggle
working folders or attached read-only inputs. Before training it checks
the prepared input checksum, split counts, continual-only preprocessor
hash, Task 1 source and configuration, actual `best.pt` hash and global
IDs 0–4, replay buffer shape/class balance/original IDs, training-only
split codes, and exact frozen transformation of the 500 replay features.
The eight-class static checkpoint/state are not loaded. These checks are
planned for Kaggle; the large prepared arrays were not locally available.

The only conditions are clean replay, targeted20 unfiltered, and the
**same** targeted20 buffer after gating. For each arm, the notebook
independently loads the Task 1 best checkpoint, seeds expansion to global
IDs 0–7 identically, hashes and compares expanded initial weights, then
trains from 70,576 class-balanced draws per epoch. Model settings remain
hidden64, heads4, layers2, MLP128, batch256, lr0.001, seed42, mixed
precision off, at most 12 epochs and patience5. The same **full clean
seen-class validation** selects each best checkpoint by loss. The reused
durable trainer records separate per-epoch `latest.pt`/`best.pt`, optimizer
and RNG for exact epoch-boundary resume; per-arm identities include the
supplied replay-label and feature hashes. If interrupted, completed arm
folders can be reused. Equal work *per epoch* does not imply equal total
work: early stopping can produce different epoch/step counts, which are
reported rather than hidden.

After selection, each arm's best model runs bounded inference on old and
new test partitions once each. Their eight-class confusion matrices sum
to combined metrics without a third inference. New pure
`summarize_task2_counts` compares against the saved five-class Task 1
reference, reporting old/new/combined accuracy, macro-F1/balanced
accuracy (focused old/new balanced averages over five/three actual classes),
per-class precision/recall/F1/support, Benign FPR,
old-attack-to-Benign errors, and signed old accuracy/fixed-five-class F1
forgetting. An old row predicted as a new class stays an error. For
example, if old accuracy falls from 0.99 to 0.70, forgetting is +0.29;
if it rises to 0.995, forgetting is -0.005. No test value selects an arm
or changes the gate. Root comparison/audit files, condition-specific
metrics/histories/checkpoints/reload checks, normalized confusion plots,
and a forgetting/per-class recall/F1 chart are planned outputs only.

**Verified synthetic:** `python3 -m unittest discover -s tests -p
'test_full_continual_metrics.py' -v` passed 2 tests for fixed old-class
averaging, new-class errors, signed forgetting, and incompatible-count
rejection. Notebook JSON/syntax/empty-output and complete repository
checks passed: 15 cells/7 code, Markdown stages and empty outputs;
`python3 -m unittest discover -s tests -p 'test_*.py' -q` ran 57 tests
(52 passed, 5 PyTorch/opt-in skips), `python3 -m compileall -q src tests`
passed, and `git diff --check` passed. Local PyTorch/GPU and
prepared arrays were unavailable; no full Task 2 metrics have been
observed. One seed, row-level splitting without session-independence
guarantees, approximate Task-1-only medians, trusted clean validation,
and a teacher trained on clean old replay examples limit interpretation.

**How to explain this in Review 2:** Task 1 retained its five known classes
well on the full old-class test, but its confidence threshold scarcely
recognized unseen classes. We are now keeping its checkpoint fixed as the
starting point for three Task 2 runs. The attacker changes only 80 stored
old attack labels to Benign; the proposed gate rejects replay examples
that disagree with the frozen Task 1 teacher. We keep the same new-class
data and draw budget in all runs, then compare clean replay, attacked
replay, and attacked replay with filtering. We will not claim that
filtering helps until the saved Kaggle results show it.

Likely faculty questions:

- **Why 80 flips rather than 100?** The declared 20% targeted rate divides
  by 400 eligible attack replay rows; 80/500 is 16% of the whole buffer.
- **Does the gate know which rows were poisoned?** No. Those simulator IDs
  enter only the separate audit after the gate decides.
- **Why hold draws at 70,576 after filtering?** It controls per-epoch
  optimizer work while exposing the consequence of losing replay classes.
- **Are all three arms equally trained in total?** Not necessarily: the
  same validation-loss patience can stop them at different epochs.
- **Can a positive rejection rate prove a useful defense?** No; downstream
  old/new accuracy, false positives and forgetting must be checked too.

---

## Full Task 2 evidence and local demonstration (2026-10-05)

### Step 1: inspect the returned archive without trusting its code

The supplied `data/task2/results.zip` hashes to
`0c73a44cca06cb92f7c21c98652808f739df67236d58d790f5cf4b93027d832e`.
It contains the result folder plus a repository snapshot, so we did not import
or execute anything from the archive. We checked all 194 member paths,
symlink/encryption flags and compressed-data integrity, then read only the
named JSON evidence and numeric NPZ metadata with pickle disabled. The archive
remains unchanged and outside Git.

The saved Task 1-before accuracy/macro-F1/Benign-FPR are
`0.9925734439413022`/`0.981162951652861`/`0.004171892742372403`.
Clean/poisoned/filtered combined accuracy is
`0.6010047020872148`/`0.44913586603196487`/`0.6077218656509111`; macro-F1
is `0.5664584687018178`/`0.5379282266745713`/`0.5578505675817005`; Benign
FPR is `0.4989118011592611`/`0.6885997951625712`/`0.4666276411782605`.
All arms completed six epochs, selected epoch one, used 1,656 steps, and began
from the same expanded-weight hash. The gate rejected 80/80 simulated poison
rows and 42/420 clean rows, retaining 378.

Aggregate improvement is not uniform robustness. The filtered arm's DoS
recall is `0.6072884683142901`, versus `0.9605879095988322` in the poisoned
unfiltered arm, and its old-attack-to-Benign count rises from 3,641 to 6,856.
This is one seed and one fixed targeted-label threat model. The compact index
is `docs/evidence/task2_targeted_summary.json`; the original JSON in the ZIP is
authoritative.

### Step 2: bind inference to real, matching artifacts

`configs/demo_artifacts.example.json` assigns each checkpoint an identity,
expected SHA-256, fit-state path/hash/scope and explicit class names. The
eight-class clean static model is the default and can use only the static
preprocessor. The five-class Task 1 teacher can use only the continual
preprocessor. Its saved max-confidence threshold is explicitly a weak novelty
baseline and is never applied after head expansion or to the static model.

`src/demo/inference.py` reads at most 2,000 CSV rows or rejects Parquet from
metadata when it exceeds the bound. It requires the exact saved 54-column
feature order, excludes only the user-selected optional broad-label column,
applies the paired frozen state, loads the checkpoint on CPU, verifies width
and class IDs, and reports real probabilities/predictions plus optional known-
class metrics. No prediction is synthesized when PyTorch/artifacts are absent.

### Step 3: make quarantine persistent without turning release into training

The poisoning view loads the real 500-row replay NPZ with `allow_pickle=False`,
uses the existing seeded targeted flip (old attacks 1–4 to Benign 0), obtains
probabilities from the frozen Task 1 teacher, and applies the existing label-
consistency gate at the saved clean-validation cutoff. The queue at
`.local/nids-demo/quarantine.sqlite3` stores original ID, supplied label,
score, threshold, model identity, row payload, status and timestamps.

Every insertion creates a `quarantined` history row. A named reviewer can
finalize a pending row as `rejected` or `released` with a required reason;
final decisions cannot be overwritten. Release updates SQLite only. It does
not call model fitting or authorize later training. The simulator's changed-
row mask is kept out of SQLite and appears only inside the separately labelled
experiment audit. Repeating the same demo is idempotent.

### Step 4: separate saved playback from computation

`src/demo/results.py` hashes and validates the ZIP, rejects traversal paths,
symlinks, encrypted entries and oversized JSON, and reads only the expected
comparison/calibration/gate/combined-metric members. The continual UI states
that it is saved experiment playback. It displays the Task 1 baseline, all
three combined results, signed forgetting, per-class precision/recall/F1,
row-normalized confusion matrices and the gate trade-off without loading a
Task 2 checkpoint or rerunning training.

### Step 5: run and verify locally

Full setup and artifact requirements are in `docs/LOCAL_DEMO.md`. Launch with:

```bash
.venv-demo/bin/streamlit run streamlit_app.py -- \
  --config configs/demo_artifacts.example.json
```

The short teacher-only path is `scripts/demo_teacher_quarantine.py`; it also
performs inference and queue insertion only. Pure logic tests passed for schema
failures, preprocessing-scope separation, targeted gate determinism, queue
persistence/idempotence/history, archive safety and normalized confusion.
Streamlit AppTest rendered all three views without exceptions using the real
saved ZIP. PyTorch 2.11.0+cpu then loaded both hash-checked real checkpoints.
The static model used its static fit state and one actual bounded Parquet row,
returning a normalized `(1, 8)` probability array. The Task 1 teacher used its
continual fit state and all 500 supplied replay rows, returning normalized
`(500, 5)` probabilities. Applying the saved threshold to those live scores
exactly reproduced 80 poisoned rejects, 42 clean rejects and 378 retained.
The teacher CLI wrote 122 rejected rows to a temporary SQLite queue. The full
unittest suite ran 67 tests: 66 passed, and only the explicitly opt-in
real-data training smoke was skipped. Scoped `pytest -q` also passed 70 tests
and 65 subtests with that one skip. The app and CLI never call `fit`; only the
repository's existing tiny synthetic CPU fit tests ran. No research-data
training or new experiment started.

---

## Final local rehearsal and provenance-safe sample (2026-10-05)

### Step 1: recover the recorded test partition without choosing new rows

The private return contained the original 9,167,581-row Parquet and split
manifest but no `.npy` split arrays. We re-ran the existing labels-only,
seed-42 full partition generator into ignored `.local/` storage. Every class
count matched the saved manifest, and the regenerated `splits.npy` SHA-256 was
`445182b087636e68089a5efd5d9645a1cc41109940d17ee778b81c935f0d2ca3`,
identical to the recorded clean-model run. This verifies the same deterministic
row assignment rather than defining a new test set.

### Step 2: export a bounded, prediction-independent UI sample

`src/demo/sample_export.py` streams only `ClassLabel` while intersecting the
strictly increasing saved test row IDs. Independent seeded reservoirs select a
fixed limit per broad class; no checkpoint, probability or correctness is
consulted. It then materializes only those selected raw rows. The flow CSV has
the static fit state's exact 54-feature order followed by `ClassLabel`.
`source_row_id` is excluded from the upload and stored in a separate provenance
CSV. Output hashes, feature order, selection seed and the warning that sample
metrics are not full-test estimates are written to a separate ignored manifest.

The real rehearsal export contains 40 rows, five from each of the eight broad
classes. The clean static checkpoint predicted them on CPU in the Flow view.
Accuracy `0.725`, macro-F1 `0.7139496468443838` and Benign FPR `0.0` describe
only this bounded, class-balanced rehearsal sample.

### Step 3: fail early with one preflight command

`src/demo/preflight.py` and `scripts/demo_preflight.py` provide a read-only
check before the faculty demonstration. They verify configured hashes; load
both checkpoint/preprocessor pairs and compare feature width, scope and head;
validate the five-class novelty artifact and 500x54 replay; safely read all
named Task 2 arm evidence; verify Parquet schema, saved split hashes and test-
index membership; and check an existing SQLite queue or writable parent. Every
failure carries a specific fix. The real preflight passed all checks.

### Step 4: rehearse browser behavior and restart persistence

Streamlit now accepts a `--database` override, keeping the final rehearsal at
`.local/nids-demo-rehearsal-20261005/quarantine.sqlite3` separate from the
default queue. The real browser rehearsal uploaded the exported CSV, selected
`ClassLabel`, and produced 40 real predictions with optional metrics. The
teacher/gate view then inserted exactly 122 suspicious rows and exposed the
separate audit of 80/80 poison and 42/420 clean rejections, 378 retained.

The reviewer rejected one item and released a second with required reviewer/
reason history. Release changed SQLite only. After a complete Streamlit stop
and restart, the UI showed 120 pending, one rejected and one released row and
both histories. The continual view loaded the hash-verified ZIP, displayed the
three exact arms, and retained the warning about DoS recall, old-attack-to-
Benign errors and single-seed scope. No training API was called.

Commands, artifacts and the timed teaching narration are in
`docs/LOCAL_DEMO.md`, `docs/FIVE_MINUTE_DEMO.md`, and
`docs/evidence/demo_rehearsal_20261005.json`.

---

## Exploratory clean Task 2 retention notebook (2026-10-05)

### Step 1: freeze the question before opening new test results

The historical three-arm Task 2 test evidence had already been inspected, so
this is labelled an exploratory follow-up. Three observations motivate it but
are recorded only as hypotheses, not causes: uniform sampling over eight
classes gives Benign 12.5% expected exposure, the existing 500-row replay has
only 100 distinct rows per old class, and the earlier Task 2 run inherited
learning rate 0.001. The study does not retrain the 65.5-minute Task 1
foundation and does not alter its fixed splits or continual preprocessor.

Only two clean configurations are allowed. A keeps the existing 500-row replay
and uniform sampler at learning rate 0.0001. B uses that learning rate plus
5,000 distinct training-only replay rows per old class and fixed class quotas:
62.5% of 70,576 draws go to old classes in original Task 1 training
proportions, while 37.5% is uniform over new classes. B deliberately combines
replay-size and sampling changes, so it cannot identify which change caused an
outcome.

### Step 2: add deterministic exposure and macro-F1 selection support

`Task2ReplayView` retains its existing stochastic uniform default. An optional
probability map now uses stable largest-remainder allocation, seeded shuffling,
and within-class replacement. For B this yields exactly 44,110 old draws and
8,822 draws for each new class per epoch. History records planned and actual
class counts plus actual unique rows drawn per class; frozen replay features
still bypass preprocessing.

`train_full_disk_backed` retains validation loss as its default. The optional
`checkpoint_selection='validation_macro_f1'` records full clean-validation
macro-F1/balanced accuracy and applies both best-checkpoint selection and
patience to macro-F1. Its strict resume header includes the non-default rule,
and best/latest checkpoints record the selection value. Existing loss-selected
runs remain compatible. These source changes and the GPU runner were published
before notebook creation; the final runner source is commit
`57009efe09f48ee742576bcc5d089a71c30b4f0e`.

### Step 3: make selection auditable and test use one-shot

`scripts/run_task2_clean_retention.py` hash-checks the complete prepared-data
and Task 1 inputs, reconstructs B's 25,000-row replay only from the fixed
training partition, verifies distinct IDs and exactly one frozen preprocessing
pass, and confirms both expanded models share an initial state hash. Both use
seed 42, batch 256, 70,576 draws (276 steps) per epoch, at most eight epochs,
patience three, the original architecture, and fresh AdamW at 0.0001. Separate
latest/best checkpoints and histories support exact epoch-boundary resume.

The rule is persisted before training: maximize combined eight-class clean-
validation macro-F1; exact ties use lower validation Benign FPR, higher old-
class focused macro-F1, then stable name. Both configurations save validation
Benign FPR and old/new focused metrics. `selection_lock.json` is durable before
the Task 1-before test record or fixed Task 2 test views are opened. Only the
selected checkpoint is evaluated, once per disjoint old/new test partition;
the confusion counts are then combined without a third inference pass. Saved
outputs include old/new/combined metrics, per-class recall/F1, forgetting,
row-normalized confusion plots, training curves, sampler exposure, hashes and
the explicit 85%-accuracy target result. Minority failures remain visible.

### Step 4: pin one Kaggle notebook and state what is unverified

`notebooks/kaggle_review2_task2_clean_retention.ipynb` pins source
`57009efe09f48ee742576bcc5d089a71c30b4f0e`. It requires Kaggle Internet, a
GPU, the complete `review2_large_prepare_<id>` folder, and the complete
`full_task1_20261003T074354_763068Z` folder. It runs focused sampler,
training-only replay, durable resume/checkpoint, and continual metric tests
before invoking the runner. Generated checkpoints/results stay in a separate
`/kaggle/working/task2_clean_retention_study` folder and outside Git.

No new training or test evaluation was run locally. JSON, all four notebook
code cells, the exact source pin, empty outputs, runner syntax/help, focused
tests and the repository test suite were checked locally. Historical Tesla T4
Task 2 measurements averaged about 21.35 seconds per train-plus-full-validation
epoch; therefore the two-arm eight-epoch cap is about 5.7 measured GPU minutes,
with prior selected-partition test inference about 18.4 seconds. Allow roughly
8–12 minutes total for hashing, replay preparation, validation selection,
plots, and Kaggle overhead. This estimate is not a promised runtime or result.

---

## Frozen-B targeted replay poisoning follow-up (2026-10-05)

### Step 1: record what the clean retention result actually established

The hash-verified retention ZIP selected configuration B with combined
accuracy `0.9769527915097729`, macro-F1 `0.670591558187877`, Benign FPR
`0.010824470651101`, old accuracy `0.9874783015309689`, new accuracy
`0.02350982350982351`, and zero Infiltration recall. Both compared arms
completed four epochs and chose epoch one. This is strong old-class retention
but weak new-class acquisition, not a successful eight-class learner. The
compact evidence and archive/notebook roles are recorded in
`docs/evidence/task2_clean_retention_summary.json`.

### Step 2: make replay identity and attack accounting fail-closed

`scripts/run_task2_b_poisoning.py` consumes the original retention evidence
ZIP, prior Task2 calibration ZIP, fixed prepared arrays and Task1 foundation.
It validates archive paths and hashes, then deterministically selects 5,000
training-only examples per old class. All 25,000 original IDs and numeric
feature/label/ID hashes must equal the saved B manifest. The transformed replay
is verified against one direct frozen-preprocessor pass and saved privately as
an object-free NPZ; it is never preprocessed again.

The targeted simulator uses original clean labels only to define its audit
budget. With seed 42, source IDs 1–4 and Benign target 0, it changes exactly
4,000 of 20,000 eligible rows. The gate itself cannot receive those original
labels or the changed mask: it receives replay features, attacked supplied
labels, probabilities from the frozen Task1 checkpoint, and the previously
verified threshold. Truth is passed only afterward to the separate gate audit.
The filtered replay must still contain all five old supplied-label classes.

### Step 3: hold the configuration-B learner constant across all arms

Clean, poisoned, and poisoned-plus-filtered arms each reload Task1 best and
expand from the same seed. The initial network hash must match both other arms
and the historical B hash. No trained B checkpoint is loaded. All arms use
the same architecture, frozen preprocessing, AdamW learning rate 0.0001,
batch 256, seed 42, 70,576 draws per epoch and historical B quotas: 44,110 old
draws in original training proportions and 8,822 for each new class. Sampling
buckets always follow each arm's supplied labels. The cap is eight epochs,
patience three, and clean-validation macro-F1 selects best checkpoints.

Durable `latest.pt` holds model/optimizer/scaler/RNG/history state at every
completed epoch and repairs `best.pt` on resume. The output reports actual,
not presumed, epochs, optimizer steps, per-class draws and unique-row exposure.
All three checkpoint and validation hashes are written to `test_open_lock.json`
before test views are created.

### Step 4: evaluate once and keep aggregate claims constrained

Each selected arm checkpoint receives one bounded pass over old fixed-test rows
and one over new fixed-test rows. Confusion counts generate old/new/combined
metrics, forgetting, Benign FPR, old-attack-to-Benign errors, per-class
recall/F1 and normalized plots without a third inference pass. The clean arm
is compared numerically with uploaded B as a reproducibility control. Any
discrepancy is reported with environment and identity checks; it does not
trigger test-guided tuning.

`notebooks/kaggle_review2_task2_b_poisoning.ipynb` pins published source
`489c27f85460b193a8ac8b9aa25d9e706f004b97`, validates focused tests, locates
the two evidence ZIPs by exact hash, and invokes the runner. Generated replay,
audit arrays, checkpoints and bulk results remain private under
`/kaggle/working`. The notebook has no stored execution outputs because this
task prepared, but did not run, the research study locally.

Local validation passed 77 tests plus 65 subtests with only the opt-in real-
data training smoke skipped. Focused replay/attack/gate/sampler/checkpoint
checks, dependency-light notebook checks, compileall, notebook/evidence JSON
validation, exact source-pin verification and `git diff --check` also passed.

---

## Clean Task2 acquisition follow-up prepared (2026-10-05)

The uploaded frozen-B `results.zip` (SHA-256
`d3ce751cf99f7c2460fd2760d5c44ee290101bd4f1f06c8adba4231b63da469e`)
passed ZIP integrity and path/symlink checks; named JSON was read without
executing archived code or unpickling checkpoints. The executed uploaded
`task2-ret-poison-v2.ipynb` (SHA-256
`06752d8fd6b42af98c63b44b80ac68730ff49a807ecdeef3f15e3a86322cb613`)
contains four executed code cells and a successful pinned-source runner call.
The clean arm reproduced earlier B metrics exactly. Combined test accuracy
was `0.9769527915097729`, macro-F1 `0.670591558187877`, old accuracy
`0.9874783015309689`, new accuracy `0.02350982350982351`, and Infiltration
recall zero: **retention with inadequate acquisition**. The existing gate
rejected 3,991/4,000 poisoned and 2,165/21,000 clean replay rows; nine
poisoned rows remained. It reviews training data, not network packets, and
its threshold is unchanged.

We audited fixed class IDs (0–4 old; 5 Infiltration, 6 Webattack, 7 Portscan),
54-feature/label row alignment, one frozen continual preprocessing pass,
Task1 checkpoint expansion, new-head gradients, deterministic exposure, and
validation-only macro-F1 selection. Synthetic gradient/resume tests passed.
A bounded seed-42 training-only audit of 10,000 Benign and 10,000
Infiltration original Parquet rows found 20,000/20,000 labels aligned with
saved split/label arrays. Under the established negative/nonfinite-to-missing
rule, 4,342 Benign and 4,482 Infiltration sampled rows had missing values;
four exact feature vectors were shared across the two samples. These are
**sample findings**, not population overlap rates or proven causes of zero
recall. No blocking implementation bug was confirmed. Frozen B gave
Infiltration 8,822/70,576 draws per epoch from 66,400 training rows and
selected epoch one; inadequate exposure is a hypothesis to test.

Published source first as `d8711cf5e35b23951ad705964581ae4ef7a58fc7`.
The one unexecuted `notebooks/kaggle_review2_task2_clean_acquisition.ipynb`
pins it. Both arms start from identical seeded Task1-best expansion and the
verified 25,000 distinct training-only old replay rows. The runner checks
every replay ID and feature/label hash against the original retention
manifest and never double-preprocesses replay. Both arms use seed 42, LR
`0.0003`, batch 256, 141,152 draws (552 steps) per epoch, max 12 epochs,
patience four, and clean-validation macro-F1 checkpoint selection. Fixed
quotas allocate 40% old replay in original old-training proportions, 40%
Infiltration, 10% Webattack and 10% Portscan; actual draws and unique rows
are saved per epoch. A uses mean eight-class CE. B adds `0.5 × T² KL` at
`T=2` from the frozen Task1 teacher on **old replay rows and old five logits
only**, normalized by full batch size. A versus B isolates KD. Compared with
historical B, revised sampling, exposure and LR are a combined intervention.

Durable best/latest checkpoints restore optimizer, scaler, RNG, history and
patience at completed epoch boundaries, with KD settings in the strict
resume header. Select the configuration by highest clean-validation combined
macro-F1; exact ties use lower Benign FPR, higher Infiltration recall, then
name. A selection lock precedes fixed-test access. Only the selected
checkpoint is tested once. Outputs include all class recall/F1, old/new/
combined metrics, forgetting, Benign FPR, normalized confusions, exposure,
curves and provenance. Report whether Infiltration improves and retention
deteriorates. High accuracy with failed new classes remains inadequate.
Historical test results were already inspected, so this is exploratory.

Local validation: 82 tests passed, one opt-in real-data smoke skipped, 65
subtests passed. No research-data training was launched. Uploaded clean-B
timing was 97.800 seconds over four train-plus-full-validation epochs, or
24.45 seconds/epoch at 70,576 draws. Doubling draws and adding teacher work
makes the 24-epoch maximum roughly 12–17 GPU minutes by extrapolation, plus
hashing, replay preparation, test inference and notebook overhead; allow
about 18–30 minutes, not a guarantee.

Final local app integration remains **pending result review**: latest and
historical playback, correctly paired inference checkpoints, live quarantine
evidence, preflight, final findings Markdown, and teacher script. No
unobserved results may be integrated; PPT/PDF files remain untouched.
