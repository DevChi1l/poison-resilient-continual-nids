"""Synthetic-only checks for full-row disk preparation and leakage boundaries."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.data import BROAD_CLASS_ORDER, DROPPED_FEATURES, fit_preprocessor
from src.data.large_data import (
    TRAIN, VALIDATION, TEST, large_data_paths, prepare_full_partitions,
    prepare_raw_feature_store, fit_disk_preprocessor, verify_full_partitions,
)
from src.training.disk_backed import DiskBackedFlowDataset


class FullDiskPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.parquet = self.root / "tiny.parquet"
        self.prepared = self.root / "prepared"
        names = [f"Feature {i}" for i in range(54)] + list(DROPPED_FEATURES)
        rows = 8 * 10
        data = {name: np.arange(rows, dtype=np.float64) + column + 1
                for column, name in enumerate(names)}
        data["Feature 1"][0] = -1
        data["Label"] = ["fine"] * rows
        data["ClassLabel"] = [BROAD_CLASS_ORDER[i // 10] for i in range(rows)]
        pq.write_table(pa.table(data), self.parquet)

    def _prepare(self):
        summary = prepare_full_partitions(self.parquet, self.prepared, seed=42, batch_size=7)
        prepare_raw_feature_store(self.parquet, self.prepared, batch_size=7)
        return summary

    def test_complete_deterministic_split_and_restarts(self):
        summary = self._prepare()
        paths = large_data_paths(self.prepared)
        self.assertEqual(summary["row_count"], 80)
        self.assertEqual(summary["counts"]["Benign"], {"train": 7, "validation": 2, "test": 1})
        self.assertEqual(summary, verify_full_partitions(paths, 80) | {
            "seed": 42, "fractions": [0.70, 0.15, 0.15],
            "split_codes": {"train": TRAIN, "validation": VALIDATION, "test": TEST}})
        first = np.load(paths.splits).copy()
        self.assertEqual(summary, prepare_full_partitions(self.parquet, self.prepared, seed=42))
        self.assertTrue(np.array_equal(first, np.load(paths.splits)))
        with self.assertRaises(ValueError):
            prepare_full_partitions(self.parquet, self.prepared, seed=43)
        self.assertTrue(np.array_equal(np.load(paths.row_ids), np.arange(80)))

    def test_feature_alignment_and_two_fit_scopes(self):
        self._prepare()
        paths = large_data_paths(self.prepared)
        raw = np.load(paths.raw_features, mmap_mode="r+")
        labels = np.load(paths.labels)
        splits = np.load(paths.splits)
        self.assertEqual(raw.shape, (80, 54))
        self.assertTrue(np.isnan(raw[0, 1]))
        self.assertEqual(float(raw[9, 0]), 10.0)
        meta = json.loads((self.prepared / "features_manifest.json").read_text())
        self.assertEqual(meta["targets_excluded"], ["Label", "ClassLabel"])
        # Test-set extremes must not enter either fit; Task 2 train enters static only.
        raw[:, 0] = 1.0
        raw[(splits == TRAIN) & (labels >= 5), 0] = 101.0
        raw[splits != TRAIN, 0] = 10_000.0
        raw.flush()
        static = fit_disk_preprocessor(self.prepared, scope="static", reservoir_size=80,
                                       seed=42, batch_size=7)
        continual = fit_disk_preprocessor(self.prepared, scope="continual", reservoir_size=80,
                                          seed=42, batch_size=7)
        self.assertEqual(float(continual.means[0]), 1.0)
        self.assertGreater(float(static.means[0]), 1.0)
        self.assertLess(float(static.means[0]), 101.0)
        self.assertEqual(float(continual.imputation_values[0]), 1.0)
        state = json.loads((self.prepared / "preprocessing_continual.json").read_text())
        self.assertEqual(state["fit_rows"], 35)
        self.assertIn("approximate", state["median_method"])
        dataset = DiskBackedFlowDataset(self.prepared, continual, partition="train",
                                        class_ids=tuple(range(5)))
        self.assertEqual(len(dataset), 35)
        ids = dataset.indices[:3]
        x, y, original = dataset.batch(ids)
        self.assertTrue(np.array_equal(y, labels[ids]))
        self.assertTrue(np.array_equal(original, ids))
        self.assertEqual(x.dtype, np.float32)
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue(np.allclose(x, continual.transform(raw[ids])))
        batches = list(dataset.iter_epoch(batch_size=8, seed=42, epoch=0))
        self.assertEqual(sum(len(batch[1]) for batch in batches), 35)
        self.assertEqual(len(set(np.concatenate([b[2] for b in batches]))), 35)
        with self.assertRaises(ValueError):
            DiskBackedFlowDataset(self.prepared, continual, partition="train",
                                  class_ids=tuple(range(5)), row_ids=np.flatnonzero(splits == TEST)[:1])
        # The existing small-array preprocessor remains usable and independent.
        old = fit_preprocessor(np.array([[1.0, -1.0], [3.0, 2.0]], dtype=np.float64),
                               ("a", "b"))
        self.assertEqual(old.transform(np.array([[2.0, -1.0]])).shape, (1, 2))

    def test_balanced_bucket_draws_are_seeded_and_epoch_specific(self):
        self._prepare()
        state = fit_disk_preprocessor(self.prepared, scope="static", reservoir_size=80)
        dataset = DiskBackedFlowDataset(self.prepared, state, partition="train",
                                        class_ids=tuple(range(8)))
        def draw(epoch):
            return np.concatenate([rows for _, _, rows in dataset.iter_epoch(
                batch_size=9, seed=42, epoch=epoch, mode="class_balanced")])
        self.assertTrue(np.array_equal(draw(0), draw(0)))
        self.assertFalse(np.array_equal(draw(0), draw(1)))
        self.assertEqual(len(draw(0)), len(dataset))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed locally")
    def test_disk_fit_cpu_checkpoint_and_early_stopping(self):
        from src.models import ModelConfig, TabularTransformerClassifier
        from src.training.disk_backed import fit_disk_backed

        self._prepare()
        state = fit_disk_preprocessor(self.prepared, scope="static", reservoir_size=80)
        train = DiskBackedFlowDataset(self.prepared, state, partition="train",
                                      class_ids=tuple(range(8)))
        val = DiskBackedFlowDataset(self.prepared, state, partition="validation",
                                    class_ids=tuple(range(8)))
        model = TabularTransformerClassifier(ModelConfig(
            num_features=54, num_classes=8, hidden_dim=8, num_heads=2,
            num_layers=1, mlp_dim=16, batch_size=8, epochs=2,
            early_stopping_patience=1, device="cpu", mixed_precision=False))
        history = fit_disk_backed(model, train, val)
        self.assertGreater(history.optimizer_steps, 0)
        self.assertIsNotNone(history.best_epoch)
        checkpoint = self.root / "tiny.pt"
        model.save(checkpoint)
        reloaded = TabularTransformerClassifier.load(checkpoint, device="cpu")
        self.assertEqual(reloaded.class_ids, model.class_ids)
