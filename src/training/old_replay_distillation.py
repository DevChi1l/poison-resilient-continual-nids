"""Optional Task-1-teacher loss on old replay rows only."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def freeze_old_teacher(teacher, student, old_class_ids: tuple[int, ...]) -> None:
    """Validate class/feature compatibility and keep the teacher immutable."""
    if (tuple(teacher.class_ids) != old_class_ids or
            tuple(student.class_ids[:len(old_class_ids)]) != old_class_ids or
            teacher.config.num_features != student.config.num_features or
            teacher.device != student.device):
        raise ValueError("Task 1 teacher and expanded student are incompatible")
    teacher.network.eval()
    teacher.network.requires_grad_(False)


def old_replay_kl(student_logits: torch.Tensor, inputs: torch.Tensor,
                  external_labels: torch.Tensor, teacher,
                  old_class_ids: tuple[int, ...], temperature: float) -> tuple[torch.Tensor, int]:
    """Return batch-normalized T² KL; new-class rows have exactly zero KD loss."""
    if temperature <= 0 or not torch.isfinite(torch.tensor(temperature)):
        raise ValueError("Distillation temperature must be positive and finite")
    mask = torch.zeros_like(external_labels, dtype=torch.bool)
    for class_id in old_class_ids:
        mask |= external_labels == class_id
    old_count = int(mask.sum().item())
    if old_count == 0:
        return student_logits.sum() * 0.0, 0
    with torch.no_grad():
        teacher_logits = teacher.network(inputs[mask]).float()
        teacher_probabilities = F.softmax(teacher_logits / temperature, dim=1)
    student_log_probabilities = F.log_softmax(
        student_logits[mask, :len(old_class_ids)].float() / temperature, dim=1
    )
    divergence = F.kl_div(student_log_probabilities, teacher_probabilities,
                          reduction="sum")
    return divergence * (temperature ** 2) / len(external_labels), old_count
