"""Deterministic training-label poisoning baselines."""

from .label_flip import PoisoningResult, apply_label_flip

__all__ = ["PoisoningResult", "apply_label_flip"]
