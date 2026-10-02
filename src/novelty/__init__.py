"""Confidence-threshold novelty baseline for model probability outputs."""

from .confidence import NoveltyResult, detect_unknown, summarize_predictions
from .calibration import ConfidenceCalibration, calibrate_confidence_threshold

__all__ = [
    "NoveltyResult",
    "detect_unknown",
    "summarize_predictions",
    "ConfidenceCalibration",
    "calibrate_confidence_threshold",
]
