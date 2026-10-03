"""Small, auditable clean-teacher label-consistency baseline."""

from .label_consistency import (
    GateResult,
    LabelConsistencyCalibration,
    apply_label_consistency_gate,
    calibrate_label_consistency,
    calibrate_label_consistency_scores,
    label_inconsistency_scores,
)

__all__ = [
    "GateResult", "LabelConsistencyCalibration", "apply_label_consistency_gate",
    "calibrate_label_consistency", "calibrate_label_consistency_scores",
    "label_inconsistency_scores",
]
