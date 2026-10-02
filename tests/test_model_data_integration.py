"""Opt-in real-data smoke test for the Developer A and B integration.

Run only with ``RUN_REAL_DATA_SMOKE=1``. The test samples eight rows from each
broad class, applies Developer B's train-only preprocessing, trains one tiny
Transformer epoch, and predicts the validation rows. It never writes processed
data or a checkpoint.
"""

from __future__ import annotations

import os
from pathlib import Path
import unittest

import numpy as np

from src.data import (
    prepare_sampled_dataset,
    sample_class_balanced_indices,
    stratified_split_indices,
)

try:
    from src.models import ModelConfig, TabularTransformerClassifier
except ModuleNotFoundError as error:  # Allows normal data-only test runs without PyTorch.
    if error.name != "torch":
        raise
    ModelConfig = None  # type: ignore[assignment,misc]
    TabularTransformerClassifier = None  # type: ignore[assignment,misc]


DATASET_PATH = Path(os.environ.get("CIC_DATASET_PATH", "data/cic-collection.parquet"))


@unittest.skipUnless(
    os.environ.get("RUN_REAL_DATA_SMOKE") == "1",
    "Set RUN_REAL_DATA_SMOKE=1 to run the real Parquet model-data smoke test.",
)
@unittest.skipUnless(DATASET_PATH.is_file(), f"Dataset not found: {DATASET_PATH}")
@unittest.skipUnless(ModelConfig is not None, "PyTorch is required for the model-data smoke test.")
class ModelDataIntegrationSmokeTest(unittest.TestCase):
    def test_preprocess_fit_and_predict(self) -> None:
        sample = sample_class_balanced_indices(
            DATASET_PATH,
            per_class_limit=8,
            seed=42,
        )
        split = stratified_split_indices(sample.labels, seed=42)
        dataset = prepare_sampled_dataset(DATASET_PATH, sample, split)

        config = ModelConfig(
            num_features=len(dataset.feature_names),
            num_classes=len(dataset.class_names),
            hidden_dim=16,
            num_heads=2,
            num_layers=1,
            mlp_dim=32,
            batch_size=32,
            epochs=1,
            device=os.environ.get("SMOKE_DEVICE", "auto"),
            mixed_precision=False,
            early_stopping_patience=None,
        )
        model = TabularTransformerClassifier(
            config,
            class_ids=tuple(sorted(dataset.class_names)),
        )
        history = model.fit(
            dataset.train.features,
            dataset.train.labels,
            dataset.validation.features,
            dataset.validation.labels,
        )
        predictions = model.predict(dataset.validation.features)

        self.assertEqual(len(dataset.feature_names), 54)
        self.assertEqual(len(dataset.class_names), 8)
        self.assertEqual(history.epochs_completed, 1)
        self.assertEqual(predictions.shape, dataset.validation.labels.shape)
        self.assertTrue(np.isin(predictions, tuple(dataset.class_names)).all())
