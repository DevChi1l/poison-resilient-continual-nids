"""Confidence-threshold novelty baseline for model probability outputs."""

from .confidence import NoveltyResult, detect_unknown, summarize_predictions

__all__ = [
    "NoveltyResult",
    "detect_unknown",
    "summarize_predictions",
]
