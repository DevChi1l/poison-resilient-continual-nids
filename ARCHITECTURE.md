# Review-2 Architecture and Interfaces

## Current architecture

The bounded data/preprocessing package, tabular Transformer classifier,
confidence-threshold novelty component, static random/targeted label flips,
known-class classification metrics, and clean two-task/replay preparation are
implemented. Kaggle T4 smoke, clean-baseline, and static poisoning comparisons
ran. The continual notebook is prepared but unexecuted; replay poisoning,
mitigation, and the one-command pipeline remain incomplete. Developers A+B+C are one owner;
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

## Canonical data contract

`src/data/` will expose a prepared dataset/task with these fields:

```python
features: numpy.ndarray       # shape (n_samples, n_features), float32
labels: numpy.ndarray         # shape (n_samples,), integer class IDs
feature_names: list[str]      # fixed feature order
class_names: dict[int, str]   # ID-to-label mapping
```

The data owner is responsible for fitting preprocessing on the training partition only and applying the same transform to validation/test partitions. Task definitions must record included classes, row counts, and seed.

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
forgetting indicates improvement. Held-out-unknown metrics remain planned.

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

## Integration contract

`run_pipeline.py` (owned by the unified team) will accept one config, invoke components in order, print only observed values, and save a timestamped JSON result manifest under `results/`. It must support a small CPU smoke configuration before a larger dataset run.
