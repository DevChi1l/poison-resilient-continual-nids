"""Model components for the Transformer-based NIDS."""

from .tabular_transformer import (
    ModelConfig,
    TabularTransformerClassifier,
    TrainingHistory,
)

__all__ = [
    "ModelConfig",
    "TabularTransformerClassifier",
    "TrainingHistory",
]
