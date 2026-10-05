"""Synthetic epoch-boundary resume and disk-view boundary checks."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.data.preprocessing import fit_preprocessor
from src.training.disk_backed import DiskBackedFlowDataset
from src.training.full_run import validate_resume_header


class ResumeHeaderTests(unittest.TestCase):
    def test_metadata_mismatch_rejected(self):
        expected = {"format_version": 1, "config": {"batch_size": 8},
                    "class_ids": [0, 3], "data_identity": {"input_sha256": "abc"},
                    "runtime": {"torch": "test"}}
        validate_resume_header(expected.copy(), expected)
        for key in ("config", "class_ids", "data_identity", "runtime"):
            altered = expected.copy()
            altered[key] = "different"
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_resume_header(altered, expected)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed locally")
class TorchResumeTests(unittest.TestCase):
    def test_epoch_boundary_resume_matches_uninterrupted(self):
        import torch
        from src.evaluation.streaming import evaluate_disk_backed
        from src.models import ModelConfig, TabularTransformerClassifier
        from src.training.full_run import train_full_disk_backed

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "prepared"
            prepared.mkdir()
            rows = 32
            raw = np.tile(np.array([[0.1, 1.0, 2.0, 3.0],
                                    [1.0, 2.0, 3.0, 4.0]], dtype=np.float32), (rows // 2, 1))
            labels = np.tile(np.array([0, 3], dtype=np.uint8), rows // 2)
            splits = np.array([0] * 20 + [1] * 8 + [2] * 4, dtype=np.uint8)
            np.save(prepared / "raw_features.npy", raw)
            np.save(prepared / "labels.npy", labels)
            np.save(prepared / "splits.npy", splits)
            np.save(prepared / "row_ids.npy", np.arange(rows, dtype=np.int64))
            feature_names = ("a", "b", "c", "d")
            (prepared / "features_manifest.json").write_text(json.dumps({
                "feature_names": list(feature_names), "row_count": rows,
            }))
            preprocessor = fit_preprocessor(raw[:20], feature_names)
            train = DiskBackedFlowDataset(prepared, preprocessor, partition="train",
                                          class_ids=(0, 3))
            val = DiskBackedFlowDataset(prepared, preprocessor, partition="validation",
                                        class_ids=(0, 3))
            test = DiskBackedFlowDataset(prepared, preprocessor, partition="test",
                                         class_ids=(0, 3))
            config = ModelConfig(num_features=4, num_classes=2, hidden_dim=8,
                                 num_heads=2, num_layers=1, mlp_dim=16,
                                 batch_size=4, epochs=3, early_stopping_patience=5,
                                 device="cpu", mixed_precision=False, seed=42)
            identity = {"input_sha256": "synthetic", "static_preprocessor_sha256": "synthetic"}
            direct = TabularTransformerClassifier(config, class_ids=(0, 3))
            direct_result = train_full_disk_backed(direct, train, val,
                                                    output_dir=root / "direct",
                                                    data_identity=identity,
                                                    progress_every_batches=100,
                                                    report=lambda _: None)
            first = TabularTransformerClassifier(config, class_ids=(0, 3))
            paused = train_full_disk_backed(first, train, val, output_dir=root / "resumed",
                                            data_identity=identity, max_epochs_this_call=1,
                                            progress_every_batches=100, report=lambda _: None)
            self.assertEqual(paused["completed_epoch"], 1)
            self.assertTrue(paused["paused"])
            with self.assertRaises(FileExistsError):
                train_full_disk_backed(first, train, val, output_dir=root / "resumed",
                                       data_identity=identity, report=lambda _: None)
            with self.assertRaises(ValueError):
                train_full_disk_backed(first, train, val, output_dir=root / "resumed",
                                       data_identity={"input_sha256": "changed"}, resume=True,
                                       report=lambda _: None)
            fresh = TabularTransformerClassifier(config, class_ids=(0, 3))
            resumed = train_full_disk_backed(fresh, train, val, output_dir=root / "resumed",
                                             data_identity=identity, resume=True,
                                             progress_every_batches=100, report=lambda _: None)
            self.assertEqual(resumed["completed_epoch"], 3)
            self.assertEqual(resumed["history"]["optimizer_steps"],
                             direct_result["history"]["optimizer_steps"])
            for a, b in zip(direct_result["history"]["epochs"], resumed["history"]["epochs"]):
                self.assertAlmostEqual(a["train_loss"], b["train_loss"], places=7)
                self.assertAlmostEqual(a["validation_loss"], b["validation_loss"], places=7)
            for name, weight in direct.network.state_dict().items():
                self.assertTrue(torch.equal(weight, fresh.network.state_dict()[name]), name)
            best = TabularTransformerClassifier.load(root / "resumed" / "best.pt", device="cpu")
            metrics = evaluate_disk_backed(best, test, batch_size=2,
                                           class_names={0: "Benign", 3: "Attack"},
                                           benign_class_id=0)
            self.assertEqual(metrics["total_rows"], 4)
            self.assertEqual(best.class_ids, (0, 3))

    def test_macro_f1_checkpoint_selection_is_recorded_and_resumable(self):
        from src.models import ModelConfig, TabularTransformerClassifier
        from src.training.full_run import train_full_disk_backed

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "prepared"
            prepared.mkdir()
            raw = np.tile(np.array([[0.1, 1.0], [1.0, 2.0]], dtype=np.float32), (12, 1))
            labels = np.tile(np.array([0, 3], dtype=np.uint8), 12)
            splits = np.array([0] * 12 + [1] * 8 + [2] * 4, dtype=np.uint8)
            np.save(prepared / "raw_features.npy", raw)
            np.save(prepared / "labels.npy", labels)
            np.save(prepared / "splits.npy", splits)
            np.save(prepared / "row_ids.npy", np.arange(24, dtype=np.int64))
            feature_names = ("a", "b")
            (prepared / "features_manifest.json").write_text(json.dumps({
                "feature_names": list(feature_names), "row_count": 24,
            }))
            preprocessor = fit_preprocessor(raw[:12], feature_names)
            train = DiskBackedFlowDataset(prepared, preprocessor, partition="train",
                                          class_ids=(0, 3))
            validation = DiskBackedFlowDataset(prepared, preprocessor,
                                               partition="validation", class_ids=(0, 3))
            config = ModelConfig(num_features=2, num_classes=2, hidden_dim=8,
                                 num_heads=2, num_layers=1, mlp_dim=16,
                                 batch_size=4, epochs=2, early_stopping_patience=3,
                                 device="cpu", mixed_precision=False, seed=42)
            identity = {"input_sha256": "synthetic", "preprocessor_sha256": "synthetic"}
            model = TabularTransformerClassifier(config, class_ids=(0, 3))
            paused = train_full_disk_backed(
                model, train, validation, output_dir=root / "macro",
                data_identity=identity, max_epochs_this_call=1,
                checkpoint_selection="validation_macro_f1", report=lambda _: None,
            )
            self.assertEqual(paused["checkpoint_selection"], "validation_macro_f1")
            self.assertIn("validation_macro_f1", paused["history"]["epochs"][0])
            fresh = TabularTransformerClassifier(config, class_ids=(0, 3))
            resumed = train_full_disk_backed(
                fresh, train, validation, output_dir=root / "macro",
                data_identity=identity, resume=True,
                checkpoint_selection="validation_macro_f1", report=lambda _: None,
            )
            self.assertEqual(resumed["completed_epoch"], 2)
            manifest = json.loads((root / "macro" / "training_manifest.json").read_text())
            self.assertEqual(manifest["checkpoint_selection"], "validation_macro_f1")
            self.assertEqual(manifest["best_selection_value"],
                             resumed["best_selection_value"])


if __name__ == "__main__":
    unittest.main()
