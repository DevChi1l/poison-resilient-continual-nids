"""Safe, read-only playback of the supplied Task 2 experiment archive."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
from stat import S_ISLNK
from typing import Any
from zipfile import BadZipFile, ZipFile

import numpy as np

from .artifacts import ArtifactConfigurationError, verify_file


ARM_ORDER = ("clean_replay", "targeted20_unfiltered", "targeted20_filtered")
ARM_LABELS = {
    "clean_replay": "Clean replay",
    "targeted20_unfiltered": "Targeted poison",
    "targeted20_filtered": "Targeted poison + gate",
}
_MAX_JSON_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class Task2Playback:
    archive_path: Path
    run_prefix: str
    comparison: dict[str, Any]
    calibration: dict[str, Any]
    gate_audit: dict[str, Any]
    combined_metrics: dict[str, dict[str, Any]]


def _validate_members(archive: ZipFile) -> None:
    for info in archive.infolist():
        normalized_name = info.filename.replace("\\", "/")
        parts = PurePosixPath(normalized_name).parts
        drive_like = bool(parts and parts[0].endswith(":"))
        if normalized_name.startswith("/") or ".." in parts or drive_like:
            raise ArtifactConfigurationError(f"Unsafe ZIP member path: {info.filename}")
        mode = (info.external_attr >> 16) & 0xFFFF
        if S_ISLNK(mode):
            raise ArtifactConfigurationError(f"ZIP symlink is not accepted: {info.filename}")
        if info.flag_bits & 0x1:
            raise ArtifactConfigurationError(f"Encrypted ZIP member is not accepted: {info.filename}")


def _read_json(archive: ZipFile, name: str) -> dict[str, Any]:
    try:
        info = archive.getinfo(name)
    except KeyError as error:
        raise ArtifactConfigurationError(f"Required result member is missing: {name}") from error
    if info.file_size > _MAX_JSON_BYTES:
        raise ArtifactConfigurationError(f"Result JSON is unexpectedly large: {name}")
    try:
        value = json.loads(archive.read(info).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactConfigurationError(f"Invalid result JSON {name}: {error}") from error
    if not isinstance(value, dict):
        raise ArtifactConfigurationError(f"Result member must contain a JSON object: {name}")
    return value


def load_task2_playback(path: str | Path, expected_sha256: str) -> Task2Playback:
    """Read only named JSON evidence; bundled source/checkpoints are never imported."""

    archive_path = Path(path)
    verify_file(archive_path, expected_sha256, label="Task 2 result archive")
    try:
        with ZipFile(archive_path) as archive:
            _validate_members(archive)
            comparison_members = [
                name for name in archive.namelist() if name.count("/") == 1 and name.endswith("/comparison.json")
            ]
            if len(comparison_members) != 1:
                raise ArtifactConfigurationError(
                    "Task 2 archive must contain exactly one top-level run comparison.json"
                )
            prefix = comparison_members[0].removesuffix("comparison.json")
            comparison = _read_json(archive, prefix + "comparison.json")
            calibration = _read_json(archive, prefix + "calibration.json")
            gate_audit = _read_json(archive, prefix + "gate_audit.json")
            combined = {
                arm: _read_json(archive, f"{prefix}{arm}/combined_test_metrics.json")
                for arm in ARM_ORDER
            }
    except BadZipFile as error:
        raise ArtifactConfigurationError(f"Invalid Task 2 ZIP: {error}") from error
    if tuple(comparison.get("conditions", ())) != ARM_ORDER:
        raise ArtifactConfigurationError("Task 2 archive has an unexpected arm order")
    if set(combined) != set(ARM_ORDER):
        raise ArtifactConfigurationError("Task 2 archive is missing a comparison arm")
    return Task2Playback(archive_path, prefix, comparison, calibration, gate_audit, combined)


def normalized_confusion(metrics: dict[str, Any]) -> np.ndarray:
    """Return row-normalized confusion values with empty rows kept at zero."""

    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ArtifactConfigurationError("Confusion matrix must be square")
    totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals != 0)


def arm_summary_rows(playback: Task2Playback) -> list[dict[str, Any]]:
    """Build compact UI rows from the saved comparison and combined metrics."""

    rows = []
    for arm in ARM_ORDER:
        details = playback.comparison["arms"][arm]
        metrics = playback.combined_metrics[arm]
        rows.append(
            {
                "Arm": ARM_LABELS[arm],
                "Combined accuracy": metrics["accuracy"],
                "Macro-F1": metrics["macro_f1"],
                "Benign FPR": metrics["benign_false_positive"]["rate"],
                "Accuracy forgetting": details["metrics"]["forgetting"]["accuracy_forgetting"],
                "Macro-F1 forgetting": details["metrics"]["forgetting"]["macro_f1_forgetting"],
                "Old attack→Benign": details["metrics"]["old_attack_to_benign"]["rate"],
                "Epochs": details["training"]["completed_epoch"],
                "Best epoch": details["training"]["best_epoch"],
                "Steps": details["training"]["optimizer_steps"],
            }
        )
    return rows
