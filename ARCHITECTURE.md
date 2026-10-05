# Review-2 Architecture and Interfaces

## Current architecture

The bounded data/preprocessing package, tabular Transformer classifier,
confidence-threshold novelty component, static random/targeted label flips,
known-class classification metrics, clean two-task/replay preparation, and a
simple frozen-teacher replay-label gate are implemented. Kaggle T4 smoke,
clean-baseline, static poisoning, clean continual, replay-label mitigation,
balanced replay exposure, and held-out confidence novelty comparisons ran
on bounded subsets. Full-row preparation, subset throughput measurement,
and a full clean static eight-class run completed on Kaggle. The
five-class full Task 1 foundation and the full three-arm Task 2 targeted replay
comparison also ran. A local artifact-backed demonstration is implemented;
the one-command training pipeline remains incomplete. Developers A+B+C are one owner;
interface changes still must be documented before integration.

```text
Unified team: data + task stream
    PreparedDataset / IncrementalTask
              |
              v
Unified team: Transformer model <---- future replay batches
    predicted labels + probabilities
              |
              +--> unified team: novelty, poisoning, metrics
              |
              v
Unified team: one-command integration and saved result manifest
```

## Local demonstration boundary

`streamlit_app.py` uses `src/demo/` as an adapter layer and does not change the
model, preprocessing, poisoning, mitigation, or evaluation algorithms.
`configs/demo_artifacts.example.json` binds each checkpoint to an expected
SHA-256, preprocessing SHA-256/scope, and explicit class mapping. Static
eight-class inference accepts only the static fit state. The five-class Task 1
teacher accepts only the continual fit state; only this model may use its saved
max-confidence novelty cutoff.

Uploaded CSV/Parquet is memory-only and capped before inference. Feature
columns must exactly match the saved 54-name order. `src/demo/quarantine.py`
persists gate-rejected rows to an ignored SQLite database with model identity,
supplied label, score, threshold, payload, status, and append-only decision
history. A release decision does not call training. Simulator ground-truth
attack masks are returned only in a separate experiment-audit object and are
never written to the operational queue.

`src/demo/results.py` validates the original Task 2 ZIP hash and member paths,
then reads only named JSON evidence. It never imports the repository snapshot
or unpickles archived checkpoints. The continual view is therefore saved
experiment playback, not an execution path.

## Canonical data contract

`src/data/` will expose a prepared dataset/task with these fields:

```python
features: numpy.ndarray       # shape (n_samples, n_features), float32
labels: numpy.ndarray         # shape (n_samples,), integer class IDs
feature_names: list[str]      # fixed feature order
class_names: dict[int, str]   # ID-to-label mapping
```

The data owner is responsible for fitting preprocessing on the training partition only and applying the same transform to validation/test partitions. Task definitions must record included classes, row counts, and seed.

### Full-row disk preparation interface

The optional large-data path in `src/data/large_data.py` does not replace the
small-array contract. `prepare_full_partitions(path, output_dir, seed=42)`
persists global `uint8` IDs 0–7, one split code per original row, explicit
original row IDs and split-index files, then verifies total coverage and
per-class counts. `prepare_raw_feature_store()` scans bounded Parquet batches
and persists one `(rows, 54)` `float32` row-aligned feature memmap; neither
`Label` nor `ClassLabel` is a feature. Negative/non-finite values are missing.
Completed stages have manifest markers for restart; partial stages are
rewritten. Input checksum verification is required by the Kaggle notebook.

`fit_disk_preprocessor(root, scope='static'|'continual', ...)` takes seeded,
bounded training-only median samples and streams imputed population moments.
Static scope is all training classes; continual scope is only Task 1 training
IDs 0–4. Validation/test never fit, and both states are separately persisted.
The medians are approximate from a recorded reservoir, unlike the exact
small-array median; the 54-feature order and invalid-value rules are retained.

`src/training/disk_backed.py` exposes `DiskBackedFlowDataset` and
`fit_disk_backed(model, train, validation=None, ...)`. A partition/class view
stores integer positions, reads and transforms only each batch, and keeps
the existing model/checkpoint/class-ID format and validation-loss early
stopping. Seeded shuffled order is the default. Optional balanced draws use
class buckets rather than a per-row float64 weight vector. This path is
prepared for future large training; the first notebook only benchmarks a
training-only subset, without opening test metrics.

For full static training, `src/training/full_run.py` adds
`train_full_disk_backed(model, train, validation, output_dir, data_identity,
resume=False, ...)`. It uses full bounded training and validation views and
atomically saves an authoritative `latest.pt` after each epoch, plus a
loadable validation-loss-selected `best.pt`. Latest records model, optimizer,
scaler, Python/NumPy/Torch CPU/CUDA RNG states, completed epoch, history,
best state, and patience. Resume rejects changed model configuration, class
IDs, prepared-data identity, or runtime. An interrupted epoch restarts from
its previous completed boundary; no mid-epoch resume is claimed. The
existing small-array `fit()` and preparation APIs are unchanged.

`src/evaluation/streaming.py` accumulates an eight-class confusion matrix
from bounded test batches and derives accuracy, macro-F1, balanced accuracy,
per-class precision/recall/F1/support, and Benign FPR. It retains no full
prediction vector. The full clean notebook opens the test partition only
after training has finished and the best checkpoint is fixed.

`src/evaluation/plots.py` reads **saved confusion counts**, not model weights
or test rows. It writes separate raw-count, annotated row-normalized, and
per-class recall/F1 PNGs. Rows in the normalized heatmap divide by each
true class's support; zero-support rows are shown as zeros. It never
overwrites an existing output. The static full-data raw-count plot remains
preserved, while these derived views expose minority-class weakness.

`src/continual_learning/disk_replay.py` adds `select_disk_replay_rows` for
the full-row path. It accepts only a training-partition disk view and returns
seeded unique original row positions, 100 per requested old class by default.
It does not materialize the full feature store; the caller uses
`DiskBackedFlowDataset.batch` on 500 selected positions and persists
frozen transformed replay features, labels, and original IDs. This does not
run Task 2 or consume replay in Task 1 training.

For the later full-row Task 2 comparison, `src/training/task2_replay.py`
exposes `Task2ReplayView(task2_view, replay_features, supplied_labels,
replay_original_ids, draws_per_epoch=70576)`. The new-class training rows
remain disk-backed and are transformed once on demand; the frozen old replay
features are already transformed and are copied into bounded batches without
another preprocessing pass. Every epoch samples uniformly across present
supplied classes, then uniformly within class, with replacement and a seed
that advances by epoch. All conditions use exactly 70,576 draws regardless
of filtering; stored replay still has at most 500 rows. The view reports
actual class draws and unique replay examples. The existing durable trainer
records this optional exposure report per epoch without changing its
ordinary disk-view path or checkpoint/resume contract.

For full clean seen-class validation, the teacher's old-class validation
probabilities are scored batchwise via `label_inconsistency_scores`; the
backward-compatible `calibrate_label_consistency_scores` accepts only the
resulting one-dimensional clean scores and returns the same 95th-percentile
threshold as the original matrix API. This avoids retaining a full
validation-probability matrix. The gate still sees supplied labels only;
clean originals and simulator changed IDs go to a separate audit. A pure
`summarize_task2_counts` combines bounded old/new confusion matrices,
keeps fixed old-class F1 and signed forgetting, and counts predictions into
new classes as old-class errors. Old/new focused balanced accuracy averages
only the five/three actually evaluated classes; combined balanced accuracy
averages all eight.

## Model contract

`src/models/` will provide a classifier with:

```python
fit(X_train, y_train, X_val=None, y_val=None, replay=None, verbose=False) -> TrainingHistory
predict(X) -> numpy.ndarray                 # integer class IDs
predict_proba(X) -> numpy.ndarray           # shape (n_samples, n_known_classes)
save(path) -> None
load(path) -> Classifier
```

`predict_proba` columns must stay aligned with the exposed `class_ids` property. The model owner must support a tiny CPU smoke run before full training.
The optional `verbose=True` prints per-epoch observed training and validation
values; it does not change validation-loss early stopping or best-state
restoration.

For the optional clean replay-exposure comparison, `ModelConfig` adds
`training_sampler='shuffled'|'class_balanced'` (default `shuffled`) and an
optional `sampler_seed` (defaulting to the model seed). Balanced mode assigns
each combined Task 2 + replay row inverse-frequency weight from its supplied
class label, then uses a seeded PyTorch `WeightedRandomSampler` with
replacement and exactly one draw per combined row per epoch. It does not
enlarge the stored replay buffer. Training history exposes expected
class probabilities, actual per-epoch counts, unique replay examples drawn,
draw/optimizer-step counts, and sampler generator-state hashes. The ordinary
shuffled path remains the default and must not supply a conflicting sampler.

## Novelty contract

`src/novelty/` will consume model probabilities only:

```python
detect_unknown(probabilities, threshold) -> numpy.ndarray  # bool mask
```

The Review-2 baseline is `max(probabilities) < threshold`. The function must not modify the model or input arrays.

The implemented optional helper is:

```python
summarize_predictions(probabilities, class_ids, threshold) -> NoveltyResult
```

It returns predicted external class IDs, maximum probabilities, and the boolean
unknown mask. `class_ids` is mandatory because probability column positions are
not external class IDs. Probability rows are validated as finite values in
`[0, 1]` whose sums are approximately one. An empty `(0, n_classes)` batch is
valid. Confidence equal to the threshold is known.
The held-out protocol calibrates the fifth percentile of maximum
probability on Task 1 **validation** rows only, before looking at Task 2
rows. The reusable `src.evaluation.novelty_metrics` reports unknown
precision/recall/F1, known false rejection, and per-unknown-class recall on
Task 1 known-test and Task 2 held-out-test rows. This is evaluation of the
confidence baseline, not a promise that it works.

## Poisoning contract

`src/poisoning/` exposes:

```python
apply_label_flip(labels, rate, seed, allowed_classes=None) -> PoisoningResult
```

Optional keyword fields are `mode='random'|'targeted'`, `source_class_ids`,
`target_class_id`, `allowed_class_ids`, and `original_row_indices`. Random
rate divides by all training rows; targeted rate divides by eligible source
rows excluding the target. Flip count is `floor(rate * eligible_count + 0.5)`.
The result contains copied original/poisoned labels, changed training
positions and optional original dataset row indices, requested/achieved
rates, eligible/total counts, and seed. Inputs are preserved.

## Evaluation contract

`src/evaluation/` exposes a JSON-serializable known-class metric helper:

```python
classification_metrics(y_true, y_pred, *, class_ids=None, class_names=None,
                       benign_class_id=None, source_class_ids=None,
                       target_class_id=None) -> dict
```

It returns accuracy, macro-F1, per-class precision/recall/F1, and confusion
counts, with optional benign false-positive and source-to-target error rates.
`forgetting_metrics(old_truth, before_predictions, after_predictions, *,
old_class_ids, all_class_ids)` computes old-class before-minus-after accuracy
and macro-F1 on the identical old-class test rows. It averages the same old
classes at both times; predictions into new classes remain errors. Negative
forgetting indicates improvement. Held-out-unknown metrics are implemented
and were exercised on the bounded Task 1/Task 2 class split; their observed
recall was low and does not establish broad novelty performance.

## Clean continual-learning contract

`src/continual_learning.prepare_two_task_dataset(selected, split)` accepts
`SelectedRows` from `materialize_selected_rows` and the existing stratified
split. It preserves global class IDs 0–7: Task 1 is 0–4 (Benign, DDoS, DoS,
Botnet, Bruteforce), Task 2 is 5–7 (Infiltration, Webattack, Portscan).
It fits the existing preprocessor **only on Task 1 training rows**, then
freezes that state for both tasks and all partitions. Validation/test rows
and Task 2 training rows never fit preprocessing. Raw row indices remain in
every partition; split and class membership are validated.

`select_balanced_replay(task1_train, class_ids=(0,1,2,3,4), per_class=100,
seed=42, forbidden_row_indices=...)` samples unique Task 1 training rows
without replacement, returns copied features/labels and raw row IDs, and
exposes `as_fit_replay()` for the model's existing `(X_replay, y_replay)`
argument. The caller provides validation/test IDs as exclusions. Task 2
sequential and replay arms begin from independent reloads of the same Task 1
checkpoint and expand to 0–7 with the same seed before training.

## Replay-label consistency gate and audit

`src/mitigation/` exposes a NumPy-only score/calibration/gate interface.
For replay row `i` with supplied old-class label `y_i`, its score is
`1 - teacher.predict_proba(X_i)[i, column_of(y_i)]`; the column is found
through the frozen Task 1 teacher's explicit `class_ids`. A threshold is
calibrated once as the 95th percentile of scores on clean Task 1 validation
rows. The gate quarantines scores **strictly above** the threshold and returns
copied retained `(X_replay, y_replay)` or `None` when empty. It receives only
features, supplied labels, teacher probabilities/class mapping, and the
frozen threshold. Clean simulator labels and changed-index masks belong only
to the separate evaluation audit; they cannot repair or inform filtering.
The gate applies only to old-class replay, never to genuinely new Task 2
training rows. This is a label-consistency baseline, not a backdoor or
new-class poisoning defense.
`src/evaluation.replay_gate_metrics` separately receives the simulator's
original labels and changed indices *after* gate decisions; it reports poison
rejection, clean false rejection, retained poison fraction, and retained
counts per supplied class. Undefined poison rates in clean conditions are
`None`, not fabricated zeros.

## Integration contract

`run_pipeline.py` (owned by the unified team) will accept one config, invoke components in order, print only observed values, and save a timestamped JSON result manifest under `results/`. It must support a small CPU smoke configuration before a larger dataset run.
