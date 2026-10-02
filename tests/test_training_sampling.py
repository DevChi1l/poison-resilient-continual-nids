"""Inverse-frequency plan and optional CPU PyTorch loader behavior."""

import importlib.util
import unittest

import numpy as np

from src.training import inverse_frequency_weights


class WeightPlanTests(unittest.TestCase):
    def test_inverse_frequency_expected_probabilities_and_input_preservation(self):
        labels = np.array([5, 5, 5, 9], dtype=np.int64)
        original = labels.copy()
        plan = inverse_frequency_weights(labels)
        np.testing.assert_allclose(plan.row_weights, [1 / 3, 1 / 3, 1 / 3, 1])
        self.assertEqual(plan.class_counts, {5: 3, 9: 1})
        self.assertAlmostEqual(plan.expected_class_probabilities[5], 0.5)
        self.assertAlmostEqual(plan.expected_class_probabilities[9], 0.5)
        np.testing.assert_array_equal(labels, original)
        self.assertFalse(np.shares_memory(labels, plan.row_weights))

    def test_invalid_labels(self):
        for labels in (np.array([]), np.array([1.0, 2.0]), np.array([[1, 2]])):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                inverse_frequency_weights(labels)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch not installed locally")
class TorchSamplerTests(unittest.TestCase):
    def test_default_and_balanced_seeded_epoch_advancement(self):
        import torch
        from torch.utils.data import RandomSampler, WeightedRandomSampler
        from src.models import ModelConfig, TabularTransformerClassifier

        default = ModelConfig(num_features=2, num_classes=2, hidden_dim=8, num_heads=2,
                              num_layers=1, mlp_dim=16, batch_size=4, epochs=2,
                              device="cpu", mixed_precision=False)
        self.assertEqual(default.training_sampler, "shuffled")
        self.assertIsNone(default.sampler_seed)
        features = np.arange(24, dtype=np.float32).reshape(12, 2)
        labels = np.array([10] * 9 + [20] * 3, dtype=np.int64)
        model = TabularTransformerClassifier(default, class_ids=[10, 20])
        loader = model._make_loader(features, model._encode_labels(labels), shuffle=True)
        self.assertIsInstance(loader.sampler, RandomSampler)
        self.assertEqual(sum(len(batch[0]) for batch in loader), 12)

        balanced = ModelConfig(num_features=2, num_classes=2, hidden_dim=8, num_heads=2,
                               num_layers=1, mlp_dim=16, batch_size=4, epochs=2,
                               device="cpu", mixed_precision=False,
                               training_sampler="class_balanced", sampler_seed=42)
        plan = inverse_frequency_weights(labels)
        def make_loader():
            classifier = TabularTransformerClassifier(balanced, class_ids=[10, 20])
            return classifier._make_loader(features, classifier._encode_labels(labels),
                                           shuffle=False, sample_weights=plan.row_weights,
                                           sampler_seed=42)
        first, second = make_loader(), make_loader()
        self.assertIsInstance(first.sampler, WeightedRandomSampler)
        self.assertEqual(first.sampler.num_samples, len(labels))
        self.assertTrue(first.sampler.replacement)
        first_epoch = [int(index) for batch in first for index in batch[2]]
        first_state = first.sampler.generator.get_state().clone()
        second_epoch = [int(index) for batch in first for index in batch[2]]
        second_state = first.sampler.generator.get_state().clone()
        self.assertFalse(torch.equal(first_state, second_state))
        self.assertEqual(first_epoch, [int(index) for batch in second for index in batch[2]])
        self.assertEqual(second_epoch, [int(index) for batch in second for index in batch[2]])

    def test_balanced_fit_exposure_on_cpu_and_config_validation(self):
        from src.models import ModelConfig, TabularTransformerClassifier

        with self.assertRaises(ValueError):
            ModelConfig(num_features=2, num_classes=2, training_sampler="unknown")
        rng = np.random.default_rng(4)
        X = rng.normal(size=(16, 2)).astype(np.float32)
        y = np.array([10] * 12 + [20] * 4, dtype=np.int64)
        config = ModelConfig(num_features=2, num_classes=2, hidden_dim=8, num_heads=2,
                             num_layers=1, mlp_dim=16, batch_size=4, epochs=2,
                             device="cpu", mixed_precision=False, early_stopping_patience=None,
                             training_sampler="class_balanced", sampler_seed=42)
        history = TabularTransformerClassifier(config, class_ids=[10, 20]).fit(X, y)
        self.assertEqual(history.training_draws, 32)
        self.assertEqual(history.optimizer_steps, 8)
        self.assertEqual(sum(history.sampled_counts_by_class.values()), 32)
        self.assertEqual(len(history.sampled_counts_by_epoch), 2)
        self.assertEqual(len(set(history.sampler_state_sha256_by_epoch)), 2)
        self.assertEqual(history.expected_class_probabilities, {10: 0.5, 20: 0.5})


if __name__ == "__main__":
    unittest.main()
