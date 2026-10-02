# Review-2 Architecture and Interfaces

## Current architecture

The bounded data/preprocessing package, tabular Transformer classifier, and
confidence-threshold novelty component are implemented. Real-data Transformer
training, continual learning, poisoning, evaluation, mitigation, and the
one-command pipeline are not yet verified or implemented as noted in
`PROJECT_STATUS.md`. Developers A+B+C are one owner; interface changes still
must be documented before integration.

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

`src/poisoning/` will expose:

```python
apply_label_flip(labels, rate, seed, allowed_classes=None) -> PoisoningResult
```

The result contains changed labels, changed row indices, requested rate, achieved rate, and seed. Original labels must remain available for evaluation.

## Evaluation contract

`src/evaluation/` will expose functions that return JSON-serializable dictionaries:

```python
classification_metrics(y_true, y_pred) -> dict
continual_metrics(task1_before, task1_after, task2_score) -> dict
novelty_metrics(is_unknown_true, is_unknown_pred) -> dict
```

## Integration contract

`run_pipeline.py` (owned by the unified team) will accept one config, invoke components in order, print only observed values, and save a timestamped JSON result manifest under `results/`. It must support a small CPU smoke configuration before a larger dataset run.
