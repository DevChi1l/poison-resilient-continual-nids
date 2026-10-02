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
