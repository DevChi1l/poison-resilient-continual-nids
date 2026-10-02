import numpy as np
import pytest


torch = pytest.importorskip("torch")

from src.models import ModelConfig, TabularTransformerClassifier


def test_train_predict_expand_and_checkpoint(tmp_path):
    rng = np.random.default_rng(7)
    X_train = rng.normal(size=(24, 6)).astype(np.float32)
    y_train = np.repeat(np.array([10, 20], dtype=np.int64), 12)
    X_val = rng.normal(size=(8, 6)).astype(np.float32)
    y_val = np.repeat(np.array([10, 20], dtype=np.int64), 4)

    config = ModelConfig(
        num_features=6,
        num_classes=2,
        hidden_dim=16,
        num_heads=2,
        num_layers=1,
        mlp_dim=32,
        batch_size=8,
        epochs=2,
        device="cpu",
        mixed_precision=False,
        early_stopping_patience=None,
    )
    classifier = TabularTransformerClassifier(config, class_ids=[10, 20])
    history = classifier.fit(X_train, y_train, X_val, y_val)

    probabilities = classifier.predict_proba(X_val)
    predictions = classifier.predict(X_val)
    assert history.epochs_completed == 2
    assert probabilities.shape == (8, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(8), atol=1e-6)
    assert set(predictions).issubset({10, 20})

    classifier.add_classes([30])
    assert classifier.class_ids == (10, 20, 30)
    assert classifier.predict_proba(X_val).shape == (8, 3)

    checkpoint_path = tmp_path / "model.pt"
    classifier.save(checkpoint_path)
    loaded = TabularTransformerClassifier.load(checkpoint_path, device="cpu")
    assert loaded.class_ids == (10, 20, 30)
    np.testing.assert_allclose(
        classifier.predict_proba(X_val),
        loaded.predict_proba(X_val),
        atol=1e-6,
    )
