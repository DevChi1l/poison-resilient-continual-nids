"""No research-data training: protocol, mapping, gradients and old-only KD checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.run_task2_acquisition import DRAWS, class_probabilities
from src.training.task2_replay import Task2ReplayView


class FakeTask2:
    """Minimal local fixture; do not depend on import resolution of `tests`."""

    partition = "train"
    class_ids = (5, 6, 7)

    def __init__(self):
        self.features = np.arange(24, dtype=np.float32).reshape(12, 2)
        self.labels = np.array([0, 1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 0])
        self.splits = np.array([0] * 11 + [2])
        self.indices = np.arange(5, 11, dtype=np.int64)

    def batch(self, rows):
        return self.features[rows] + 100, self.labels[rows].copy(), rows.copy()


class AcquisitionProtocolTests(unittest.TestCase):
    def test_exact_quota_and_training_only_rows(self):
        order = tuple(str(i) for i in range(8))
        split = {"counts": {str(i): {"train": (i + 1) * 100} for i in range(5)}}
        probabilities = class_probabilities(split, order)
        self.assertAlmostEqual(sum(probabilities[i] for i in range(5)), 0.4)
        self.assertEqual([probabilities[i] for i in (5, 6, 7)], [0.4, 0.1, 0.1])
        fake = FakeTask2()
        X = np.arange(10, dtype=np.float32).reshape(5, 2)
        y = np.arange(5, dtype=np.int64)
        row_ids = np.arange(5, dtype=np.int64)
        view = Task2ReplayView(fake, X, y, row_ids, draws_per_epoch=256,
                               class_draw_probabilities=probabilities)
        self.assertEqual(sum(view.planned_class_draw_counts.values()), 256)
        self.assertGreater(view.planned_class_draw_counts[5],
                           view.planned_class_draw_counts[6])
        self.assertEqual(DRAWS, 141_152)
        list(view.iter_epoch(batch_size=64, seed=42, epoch=0))
        self.assertEqual(view.last_epoch_exposure["sampled_counts"],
                         view.last_epoch_exposure["planned_class_draw_counts"])


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch absent")
class DistillationTests(unittest.TestCase):
    def test_head_mapping_new_gradient_and_teacher_freeze(self):
        import torch
        from src.models import ModelConfig, TabularTransformerClassifier
        from src.training.old_replay_distillation import freeze_old_teacher, old_replay_kl

        config = ModelConfig(num_features=4, num_classes=5, hidden_dim=8,
            num_heads=2, num_layers=1, mlp_dim=16, dropout=0, device="cpu",
            mixed_precision=False, seed=42)
        teacher = TabularTransformerClassifier(config, class_ids=tuple(range(5)))
        student = TabularTransformerClassifier(config, class_ids=tuple(range(5)))
        before = student.network.classifier[-1].weight.detach().clone()
        student.add_classes((5, 6, 7))
        self.assertEqual(student.class_ids, tuple(range(8)))
        self.assertTrue(torch.equal(before, student.network.classifier[-1].weight[:5]))
        freeze_old_teacher(teacher, student, tuple(range(5)))
        self.assertFalse(teacher.network.training)
        self.assertFalse(any(p.requires_grad for p in teacher.network.parameters()))
        inputs = torch.randn(4, 4)
        labels = torch.tensor([0, 1, 5, 7])
        logits = student.network(inputs)
        kd, count = old_replay_kl(logits, inputs, labels, teacher, tuple(range(5)), 2.0)
        self.assertEqual(count, 2)
        expected, _ = old_replay_kl(logits[:2], inputs[:2], labels[:2], teacher,
                                    tuple(range(5)), 2.0)
        self.assertTrue(torch.allclose(kd, expected / 2))
        # KD cannot constrain new-class rows or new output columns.
        kd.backward(retain_graph=True)
        self.assertTrue(torch.equal(student.network.classifier[-1].weight.grad[5:],
                                    torch.zeros_like(student.network.classifier[-1].weight.grad[5:])))
        student.network.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(logits, labels).backward()
        self.assertGreater(student.network.classifier[-1].weight.grad[5:].abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in teacher.network.parameters()))

    def test_no_old_rows_has_zero_kd_and_resume_header_rejects_changed_kd(self):
        import torch
        from src.models import ModelConfig, TabularTransformerClassifier
        from src.training.full_run import _resume_header, validate_resume_header
        from src.training.old_replay_distillation import freeze_old_teacher, old_replay_kl
        config = ModelConfig(num_features=2, num_classes=5, hidden_dim=8, num_heads=2,
            num_layers=1, mlp_dim=16, device="cpu", mixed_precision=False)
        teacher = TabularTransformerClassifier(config)
        student = TabularTransformerClassifier(config)
        student.add_classes((5, 6, 7))
        freeze_old_teacher(teacher, student, tuple(range(5)))
        inputs = torch.randn(3, 2)
        logits = student.network(inputs)
        kd, count = old_replay_kl(logits, inputs, torch.tensor([5, 6, 7]), teacher,
                                  tuple(range(5)), 2.0)
        self.assertEqual(count, 0)
        self.assertEqual(kd.item(), 0.0)
        base = _resume_header({"learning_rate": 0.0003}, tuple(range(8)),
                              {"input": "synthetic"}, {"torch": "synthetic"},
                              "validation_macro_f1")
        saved = {**base, "distillation": {"weight": 0.5, "temperature": 2.0}}
        validate_resume_header(saved, saved)
        with self.assertRaises(ValueError):
            validate_resume_header(saved, {**base, "distillation": {"weight": 0.2}})

    def test_durable_kd_resume_rejects_protocol_change(self):
        import torch
        from src.models import ModelConfig, TabularTransformerClassifier
        from src.training.full_run import train_full_disk_backed

        class TinyView:
            def __init__(self, partition):
                self.partition = partition
                self.class_ids = tuple(range(8))
                self.features = np.zeros((8, 2), dtype=np.float32)
                self.indices = np.arange(8)

            def __len__(self):
                return 8

            def batch(self, ids):
                labels = np.asarray(ids, dtype=np.int64)
                X = np.stack((labels / 7, 1 - labels / 7), axis=1).astype(np.float32)
                return X, labels, labels

            def iter_epoch(self, *, batch_size, seed, epoch, mode):
                for start in range(0, 8, batch_size):
                    yield self.batch(self.indices[start:start + batch_size])

        config = ModelConfig(num_features=2, num_classes=5, hidden_dim=8,
            num_heads=2, num_layers=1, mlp_dim=16, dropout=0, batch_size=4,
            epochs=2, device="cpu", mixed_precision=False,
            training_sampler="class_balanced", early_stopping_patience=3)
        teacher = TabularTransformerClassifier(config)
        student = TabularTransformerClassifier(config)
        student.add_classes((5, 6, 7))
        train = TinyView("train")
        val = TinyView("validation")
        with tempfile.TemporaryDirectory() as directory:
            kwargs = {"output_dir": directory, "data_identity": {"synthetic": True},
                      "checkpoint_selection": "validation_macro_f1",
                      "distillation_teacher": teacher,
                      "distillation_old_class_ids": tuple(range(5)),
                      "distillation_weight": 0.5, "distillation_temperature": 2.0,
                      "report": lambda _: None}
            paused = train_full_disk_backed(student, train, val,
                                            max_epochs_this_call=1, **kwargs)
            self.assertTrue(paused["paused"])
            self.assertGreater(paused["history"]["epochs"][0]["distillation_old_rows"], 0)
            with self.assertRaises(ValueError):
                train_full_disk_backed(student, train, val, resume=True,
                                       **{**kwargs, "distillation_weight": 0.2})
            fresh = TabularTransformerClassifier(config)
            fresh.add_classes((5, 6, 7))
            completed = train_full_disk_backed(fresh, train, val, resume=True, **kwargs)
            self.assertEqual(completed["completed_epoch"], 2)
            self.assertEqual(completed["history"]["optimizer_steps"], 4)
            self.assertTrue((Path(directory) / "best.pt").is_file())


if __name__ == "__main__":
    unittest.main()
