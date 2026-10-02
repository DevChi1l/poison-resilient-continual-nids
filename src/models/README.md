# Model / NIDS Component

Owner: **Unified A+B+C team**. The implementation originated on
`feature/model-nids` and is now a shared team responsibility.

`tabular_transformer.py` implements a numeric feature-token Transformer. Each
preprocessed flow feature becomes a learned token, a Transformer encoder models
feature interactions, and the CLS token is classified into attack-class IDs.

## Public API

```python
from src.models import ModelConfig, TabularTransformerClassifier

config = ModelConfig(
    num_features=number_of_preprocessed_features,
    num_classes=number_of_known_classes,
    device="auto",           # CUDA when available, CPU otherwise
    mixed_precision=True,    # active only on CUDA
)
model = TabularTransformerClassifier(config, class_ids=known_class_ids)
history = model.fit(X_train, y_train, X_val, y_val, replay=None, verbose=True)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
model.save("checkpoints/review2_model.pt")
```

Inputs must follow the contract in `ARCHITECTURE.md`: finite `float32` features
with shape `(n_samples, n_features)` and one-dimensional integer class IDs.
`predict_proba` columns always align with `model.class_ids`.
`verbose=True` prints observed train loss, validation loss, and validation
accuracy after each epoch. Validation loss still selects and restores the best
model state before `fit` returns; the default `verbose=False` preserves quiet
existing callers. `ModelConfig(training_sampler="class_balanced",
sampler_seed=42)` optionally draws combined Task 2 + replay rows with
inverse-frequency class weights and replacement. The default remains
`training_sampler="shuffled"`. `history.to_dict()` records the sampler seed,
expected class probabilities, actual per-epoch class counts, unique replay
examples drawn, training draws, and optimizer steps. Balanced sampling changes
exposure, not the number of stored replay examples.

## Continual-learning support

Call `add_classes(new_class_ids)` before a task containing unseen labels, or
let `fit` discover those labels. The classifier expands its output layer while
retaining existing output weights. Replay data is passed as
`replay=(X_replay, y_replay)`.

## Compute modes

- **Local smoke test:** set a small model (`hidden_dim=32`, `num_layers=1`), a
  small batch, `device="cpu"`, and `mixed_precision=False`.
- **Colab/Lightning experiment:** keep `device="auto"`, set an appropriate GPU
  batch size, and use the default CUDA mixed precision. Epochs, batch size,
  architecture size, workers, learning rate, dropout, seed, and early stopping
  are all configurable in `ModelConfig`.

PyTorch is required but is not installed in the current system environment.
The unified team owns the shared dependency manifest. This module has a
torch-gated smoke test that will run once PyTorch is available.
