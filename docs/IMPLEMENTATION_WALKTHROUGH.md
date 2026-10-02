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
