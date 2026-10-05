#!/usr/bin/env python3
"""Run the frozen-configuration-B replay poisoning comparison on Kaggle.

The three arms differ only in the replay labels/filtering applied before the
fixed configuration-B sampler.  Task 1 is never retrained.  Test partitions
are opened only after every arm has a validation-selected durable checkpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import random
import shutil
import stat
import subprocess
import sys
from time import perf_counter
from typing import Any
from zipfile import ZipFile

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.run_task2_clean_retention import (  # noqa: E402
    atomic_json,
    combined_partition_metrics,
    save_training_curves,
    sha256_array,
    sha256_file,
    state_hash,
    write_or_match,
)


EXPECTED_PREP_SOURCE_SHA = "5c0d519596bc6c4b99948126443635c2f7eb15f1"
EXPECTED_TASK1_SOURCE_SHA = "e6ebfcf7a9beec4d2b8fd5245e7c62314b9fadf9"
EXPECTED_INPUT_SHA256 = "666af53c788c79312b421c607289cb555c2335ae51ba39b2700a357595fa8dba"
EXPECTED_TASK1_BEST_SHA256 = "aed830e152d61525c641a95ac8eded661caa8c318d9aed910abb954782454de6"
EXPECTED_RETENTION_ZIP_SHA256 = "1efb433f2f77501a290600a67a7f4e0b1f760b0f5b42b983db49ed1ffd2cacf0"
EXPECTED_PRIOR_TASK2_ZIP_SHA256 = "0c73a44cca06cb92f7c21c98652808f739df67236d58d790f5cf4b93027d832e"
EXPECTED_CALIBRATION_JSON_SHA256 = "13e7700380da0a6fbf3dc7ad5a3cec9a18dca74c471d6dbff733ffa0b58bae46"
RETENTION_ROOT = "task2_clean_retention_study/"
CALIBRATION_MEMBER = (
    "full_task2_targeted_20261003T095508_992113Z/calibration.json"
)
# SHA-256 of the exact JSON bytes read by this runner, derived from the two
# original, whole-archive-hash-verified ZIPs. Folder inputs use these same bytes.
RETENTION_EVIDENCE_SHA256 = {
    "protocol.json": "42b7354799c470b0b9b252763f9ed080ca2f6695af576224d4fc93eba438e884",
    "large_replay_manifest.json": "e7187cd1054e5229a937a4a49968a64e3848434d71affb1492b334dc7438c6f9",
    "B_large_proportional_lr1e4/arm_config.json": "d68bf9ba1e7609a852b1b081474f6a39244569c8432374706e86dcdc78b528b8",
    "B_large_proportional_lr1e4/arm_summary.json": "25c67a6fa8d1f7b778a735909430d80fabf92ffc3c256d147bfaaf8dbfcda465",
    "B_large_proportional_lr1e4/validation_metrics.json": "9fd26a52aaf69dc9d096e8fff0bdb60e4b791f0ca4d1e2f96fe2a1c337d03c26",
    "selected_test_summary.json": "fe355f64316da7e18c6a5c9bc2a5a6e7115fb7c5b76aecf93468cd4c44a59d93",
    "study_manifest.json": "2f7cd365f5309ada3b547313f50853dd1321fc55e069e73cf5785ccd01418d26",
    "environment.json": "b11a3dfbda3a2ea492ea59b20152b73f9bc6ff904a81e7e90094ffaa8ade3f37",
}
PRIOR_EVIDENCE_SHA256 = {"calibration.json": EXPECTED_CALIBRATION_JSON_SHA256}
HISTORICAL_RETENTION_SOURCE = "57009efe09f48ee742576bcc5d089a71c30b4f0e"
HISTORICAL_B_INITIAL_SHA256 = "e53fb3c899176c99cb4914a8dd2f03f3addba355e4b3566a5dd17df23853640a"
OLD_IDS = tuple(range(5))
NEW_IDS = (5, 6, 7)
ALL_IDS = tuple(range(8))
ARMS = ("clean_B", "targeted20_B", "targeted20_filtered_B")
DRAW_COUNT = 70_576
SEED = 42
ATTACK_RATE = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen configuration-B clean/poisoned/filtered Kaggle comparison."
    )
    parser.add_argument("--prepared-dir", required=True, type=Path)
    parser.add_argument("--task1-run-dir", required=True, type=Path)
    retention = parser.add_mutually_exclusive_group(required=True)
    retention.add_argument("--retention-results-zip", type=Path)
    retention.add_argument("--retention-results-dir", type=Path)
    prior = parser.add_mutually_exclusive_group(required=True)
    prior.add_argument("--prior-task2-results-zip", type=Path)
    prior.add_argument("--prior-task2-results-dir", type=Path)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output-dir", type=Path)
    destination.add_argument("--resume-run-dir", type=Path)
    parser.add_argument("--max-epochs-this-call", type=int)
    return parser.parse_args()


def require_kaggle_directory(path: Path, label: str) -> Path:
    resolved = path.resolve()
    roots = (Path("/kaggle/input"), Path("/kaggle/working"))
    if not resolved.is_dir() or not any(resolved.is_relative_to(root) for root in roots):
        raise ValueError(f"{label} must be an existing /kaggle/input or /kaggle/working folder")
    return resolved


def require_kaggle_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    roots = (Path("/kaggle/input"), Path("/kaggle/working"))
    if not resolved.is_file() or not any(resolved.is_relative_to(root) for root in roots):
        raise ValueError(f"{label} must be an existing /kaggle/input or /kaggle/working file")
    return resolved


def output_folder(args: argparse.Namespace) -> Path:
    if args.resume_run_dir is not None:
        previous = require_kaggle_directory(args.resume_run_dir, "resume run")
        if previous.is_relative_to(Path("/kaggle/working")):
            return previous
        destination = Path("/kaggle/working") / (
            "task2_b_poisoning_resume_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        )
        shutil.copytree(previous, destination)
        return destination
    if args.output_dir is not None:
        destination = args.output_dir.resolve()
        if not destination.is_relative_to(Path("/kaggle/working")):
            raise ValueError("output-dir must be under /kaggle/working")
        destination.mkdir(parents=True, exist_ok=True)
        return destination
    destination = Path("/kaggle/working") / (
        "task2_b_poisoning_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    )
    destination.mkdir(exist_ok=False)
    return destination


def _validate_archive_members(archive: ZipFile) -> None:
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        if (path.is_absolute() or ".." in path.parts or info.flag_bits & 0x1 or
                stat.S_ISLNK(mode)):
            raise ValueError(f"Unsafe ZIP member rejected: {info.filename!r}")


def read_verified_json_archive(
    path: Path,
    *,
    expected_sha256: str,
    members: tuple[str, ...],
    member_sha256: dict[str, str] | None = None,
) -> dict[str, dict]:
    """Read only named JSON evidence from a hash-pinned, structurally safe ZIP."""

    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"Evidence ZIP hash mismatch for {path}: {actual}")
    records: dict[str, dict] = {}
    with ZipFile(path) as archive:
        _validate_archive_members(archive)
        names = set(archive.namelist())
        missing = set(members) - names
        if missing:
            raise ValueError(f"Evidence ZIP lacks required members: {sorted(missing)}")
        for member in members:
            payload = archive.read(member)
            if member_sha256 is not None and hashlib.sha256(payload).hexdigest() != member_sha256[member]:
                raise ValueError(f"Evidence file hash mismatch: {member}")
            records[member] = json.loads(payload.decode("utf-8"))
    return records


def read_verified_json_folder(
    folder: Path, *, root: str, member_sha256: dict[str, str]
) -> dict[str, dict]:
    """Read only pinned JSON files from an extracted study root, with no symlinks."""

    if not folder.is_dir() or folder.is_symlink():
        raise ValueError(f"Evidence folder must be a real directory: {folder}")
    records: dict[str, dict] = {}
    for relative, expected_hash in member_sha256.items():
        part = PurePosixPath(relative)
        if part.is_absolute() or ".." in part.parts or not relative.endswith(".json"):
            raise ValueError(f"Unsafe evidence filename: {relative!r}")
        target = folder.joinpath(*part.parts)
        current = folder
        for component in part.parts:
            current = current / component
            if current.is_symlink():
                raise ValueError(f"Evidence symlink rejected: {relative}")
        if not target.is_file():
            raise FileNotFoundError(f"Required evidence file missing: {target}")
        payload = target.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_hash:
            raise ValueError(f"Evidence file hash mismatch: {target}")
        records[root + relative] = json.loads(payload.decode("utf-8"))
    return records


def read_verified_evidence(
    source: Path, *, root: str, zip_sha256: str, member_sha256: dict[str, str]
) -> dict[str, dict]:
    if source.is_dir():
        return read_verified_json_folder(source, root=root, member_sha256=member_sha256)
    if source.is_file():
        members = tuple(root + name for name in member_sha256)
        return read_verified_json_archive(
            source,
            expected_sha256=zip_sha256,
            members=members,
            member_sha256={root + name: digest for name, digest in member_sha256.items()},
        )
    raise FileNotFoundError(f"Evidence ZIP or folder missing: {source}")


def require_supplied_class_buckets(
    labels: np.ndarray, required_ids: tuple[int, ...] = OLD_IDS
) -> dict[str, int]:
    """Return supplied-label counts, failing before training if any bucket is empty."""

    values = np.asarray(labels)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Replay supplied labels must be a one-dimensional integer array")
    counts = {str(class_id): int(np.count_nonzero(values == class_id))
              for class_id in required_ids}
    empty = [class_id for class_id in required_ids if counts[str(class_id)] == 0]
    if empty:
        raise RuntimeError(
            "Filtering emptied required supplied-label replay bucket(s): "
            + ", ".join(str(value) for value in empty)
        )
    return counts


def fixed_b_probabilities(split_manifest: dict, class_order: tuple[str, ...]) -> dict[int, float]:
    counts = {
        class_id: int(split_manifest["counts"][class_order[class_id]]["train"])
        for class_id in OLD_IDS
    }
    total = sum(counts.values())
    probabilities = {
        class_id: 0.625 * counts[class_id] / total for class_id in OLD_IDS
    }
    probabilities.update({class_id: 0.125 for class_id in NEW_IDS})
    if not np.isclose(sum(probabilities.values()), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("Frozen configuration-B probabilities do not sum to one")
    return probabilities


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _load_or_write_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(
                not np.array_equal(saved[key], value) for key, value in arrays.items()
            ):
                raise ValueError(f"Existing private NPZ differs from reconstructed state: {path}")
    else:
        _atomic_npz(path, **arrays)
    return sha256_file(path)


def _evidence_bundle(retention_source: Path, prior_source: Path) -> dict[str, Any]:
    retention = read_verified_evidence(
        retention_source,
        root=RETENTION_ROOT,
        zip_sha256=EXPECTED_RETENTION_ZIP_SHA256,
        member_sha256=RETENTION_EVIDENCE_SHA256,
    )
    prior = read_verified_evidence(
        prior_source,
        root=CALIBRATION_MEMBER.removesuffix("calibration.json"),
        zip_sha256=EXPECTED_PRIOR_TASK2_ZIP_SHA256,
        member_sha256=PRIOR_EVIDENCE_SHA256,
    )
    by_short_name = {
        member.removeprefix(RETENTION_ROOT): value for member, value in retention.items()
    }
    by_short_name["prior_calibration"] = prior[CALIBRATION_MEMBER]
    return by_short_name


def _verify_retention_evidence(evidence: dict[str, Any]) -> None:
    protocol = evidence["protocol.json"]
    replay = evidence["large_replay_manifest.json"]
    config = evidence["B_large_proportional_lr1e4/arm_config.json"]
    summary = evidence["B_large_proportional_lr1e4/arm_summary.json"]
    result = evidence["selected_test_summary.json"]
    calibration = evidence["prior_calibration"]
    if not (
        protocol["source_sha"] == HISTORICAL_RETENTION_SOURCE
        and protocol["prepared_input_sha256"] == EXPECTED_INPUT_SHA256
        and protocol["task1_best_sha256"] == EXPECTED_TASK1_BEST_SHA256
        and replay["seed"] == SEED
        and replay["source_partition"] == "fixed training split only"
        and replay["rows_per_old_class"] == 5_000
        and replay["total_rows"] == 25_000
        and config["stored_replay_rows"] == 25_000
        and config["data_identity"]["expanded_initial_sha256"]
        == HISTORICAL_B_INITIAL_SHA256
        and summary["completed_epoch"] == 4
        and summary["best_epoch"] == 1
        and result["selected_condition"] == "B_large_proportional_lr1e4"
        and np.isclose(result["combined_test"]["accuracy"], 0.9769527915097729)
        and np.isclose(result["combined_test"]["macro_f1"], 0.670591558187877)
        and np.isclose(result["benign_fpr"], 0.010824470651101)
        and np.isclose(result["old_test"]["accuracy"], 0.9874783015309689)
        and np.isclose(result["new_test"]["accuracy"], 0.02350982350982351)
        and result["combined_test"]["per_class"]["5"]["recall"] == 0.0
        and calibration["task1_best_sha256"] == EXPECTED_TASK1_BEST_SHA256
        and calibration["fit_scope"] == "clean Task 1 validation only"
        and calibration["validation_count"] == 1_360_123
        and calibration["quantile"] == 0.95
        and calibration["threshold"] == 0.006338521838188171
    ):
        raise ValueError("Retention or calibration evidence differs from the verified study")


def _evaluate_validation(model: Any, old_view: Any, new_view: Any,
                         class_names: dict[int, str]) -> dict:
    from src.evaluation.streaming import evaluate_disk_backed

    old = evaluate_disk_backed(
        model, old_view, batch_size=256, class_names=class_names, benign_class_id=0
    )
    new = evaluate_disk_backed(
        model, new_view, batch_size=256, class_names=class_names, benign_class_id=0
    )
    return combined_partition_metrics(old, new, class_names)


def _delta(left: float, right: float) -> float:
    return float(left) - float(right)


def main() -> None:
    args = parse_args()
    if args.max_epochs_this_call is not None and args.max_epochs_this_call < 1:
        raise ValueError("max-epochs-this-call must be positive")

    import torch
    from src.continual_learning.disk_replay import select_disk_replay_rows
    from src.data import BROAD_CLASS_ORDER, FittedPreprocessor
    from src.data.large_data import large_data_paths, verify_full_partitions
    from src.evaluation.plots import save_result_plots
    from src.evaluation.replay_gate import replay_gate_metrics
    from src.evaluation.streaming import evaluate_disk_backed, summarize_task2_counts
    from src.mitigation import apply_label_consistency_gate
    from src.models import TabularTransformerClassifier
    from src.poisoning import apply_label_flip
    from src.training.disk_backed import DiskBackedFlowDataset
    from src.training.full_run import train_full_disk_backed
    from src.training.task2_replay import Task2ReplayView

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; enable a Kaggle GPU and restart the session")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    prepared_dir = require_kaggle_directory(args.prepared_dir, "prepared input")
    task1_dir = require_kaggle_directory(args.task1_run_dir, "Task 1 run")
    retention_source = (
        require_kaggle_file(args.retention_results_zip, "retention evidence ZIP")
        if args.retention_results_zip is not None else
        require_kaggle_directory(args.retention_results_dir, "retention evidence folder")
    )
    prior_source = (
        require_kaggle_file(args.prior_task2_results_zip, "prior Task 2 evidence ZIP")
        if args.prior_task2_results_zip is not None else
        require_kaggle_directory(args.prior_task2_results_dir, "prior Task 2 evidence folder")
    )
    destination = output_folder(args)
    inputs = (
        prepared_dir, task1_dir,
        retention_source if retention_source.is_dir() else retention_source.parent,
        prior_source if prior_source.is_dir() else prior_source.parent,
    )
    if any(destination == value or destination.is_relative_to(value) for value in inputs):
        raise ValueError("Output must be separate from every read-only input")
    if shutil.disk_usage("/kaggle/working").free < 1_000_000_000:
        raise RuntimeError("Less than 1 GB is free under /kaggle/working")

    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_device_count": torch.cuda.device_count(),
    }
    evidence = _evidence_bundle(retention_source, prior_source)
    _verify_retention_evidence(evidence)

    prep_manifest = json.loads((prepared_dir / "manifest.json").read_text())
    split_manifest = json.loads((prepared_dir / "split_manifest.json").read_text())
    features_manifest = json.loads((prepared_dir / "features_manifest.json").read_text())
    continual_record = json.loads((prepared_dir / "preprocessing_continual.json").read_text())
    task1_final = json.loads((task1_dir / "final_manifest.json").read_text())
    task1_config = json.loads((task1_dir / "run_config.json").read_text())
    task1_training = json.loads((task1_dir / "training_manifest.json").read_text())
    task1_reload = json.loads((task1_dir / "reload_verification.json").read_text())
    paths = large_data_paths(prepared_dir)

    if not (
        prep_manifest["completed_preparation_and_benchmark_only"] is True
        and prep_manifest["source_sha"] == EXPECTED_PREP_SOURCE_SHA
        and prep_manifest["input_sha256"] == EXPECTED_INPUT_SHA256
        and prep_manifest["input_rows"] == split_manifest["row_count"]
        == features_manifest["row_count"] == 9_167_581
        and split_manifest["seed"] == SEED
        and split_manifest["class_order"] == list(BROAD_CLASS_ORDER)
        and split_manifest["split_codes"] == {"train": 0, "validation": 1, "test": 2}
        and features_manifest["dtype"] == "float32"
        and features_manifest["targets_excluded"] == ["Label", "ClassLabel"]
        and len(features_manifest["feature_names"]) == 54
        and continual_record["scope"] == "continual"
        and continual_record["fit_rows"] == 6_347_232
        and continual_record["feature_names"] == features_manifest["feature_names"]
        and continual_record["reservoir_seed"] == SEED
        and continual_record["reservoir_size_used"] == 200_000
    ):
        raise ValueError("Prepared data or frozen continual preprocessing changed")
    if verify_full_partitions(paths, 9_167_581)["counts"] != split_manifest["counts"]:
        raise ValueError("Prepared arrays do not match the fixed split manifest")

    task1_identity = task1_final["data_identity"]
    if not (
        task1_final["status"] == "completed full-data Task 1 foundation only"
        and task1_final["task2_trained"] is False
        and task1_final["source_sha"] == task1_config["source_sha"]
        == EXPECTED_TASK1_SOURCE_SHA
        and task1_final["class_ids"] == task1_config["class_ids"]
        == task1_training["class_ids"] == list(OLD_IDS)
        and task1_final["completed_epoch"] == 12
        and task1_final["best_epoch"] == 9
        and task1_final["data_identity"] == task1_config["data_identity"]
        == task1_training["data_identity"]
        and task1_identity == evidence["prior_calibration"]["data_identity"]
    ):
        raise ValueError("Task 1 run is not the verified completed foundation")
    if not (
        task1_identity["input_sha256"] == EXPECTED_INPUT_SHA256
        and task1_identity["prep_source_sha"] == EXPECTED_PREP_SOURCE_SHA
        and task1_identity["preprocessing_scope"] == "Task 1 training global IDs 0-4 only"
        and task1_identity["task1_class_ids"] == list(OLD_IDS)
    ):
        raise ValueError("Task 1 identity changed")
    for path, key in (
        (paths.raw_features, "raw_features_sha256"),
        (paths.labels, "labels_sha256"),
        (paths.splits, "splits_sha256"),
        (paths.row_ids, "row_ids_sha256"),
        (prepared_dir / "split_manifest.json", "split_manifest_sha256"),
        (prepared_dir / "preprocessing_continual.json", "continual_preprocessing_sha256"),
    ):
        if sha256_file(path) != task1_identity[key]:
            raise ValueError(f"Prepared identity mismatch: {key}")
    if sha256_file(task1_dir / "preprocessing_continual.json") != \
            task1_identity["continual_preprocessing_sha256"]:
        raise ValueError("Task 1 copy of the continual preprocessor changed")

    best_path = task1_dir / "best.pt"
    task1_best_sha = sha256_file(best_path)
    if not (
        task1_best_sha == EXPECTED_TASK1_BEST_SHA256
        == task1_final["best_checkpoint_sha256"]
        == task1_reload["best_checkpoint_sha256"]
    ):
        raise ValueError("Task 1 best checkpoint hash changed")
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    if not (
        checkpoint["class_ids"] == list(OLD_IDS)
        and checkpoint["data_identity"] == task1_identity
        and checkpoint["best_epoch"] == task1_final["best_epoch"]
    ):
        raise ValueError("Task 1 checkpoint metadata is incompatible")
    del checkpoint

    preprocessor = FittedPreprocessor(
        tuple(continual_record["feature_names"]),
        np.asarray(continual_record["imputation_values"], dtype=np.float64),
        np.asarray(continual_record["means"], dtype=np.float64),
        np.asarray(continual_record["scales"], dtype=np.float64),
    )
    labels_map = np.load(paths.labels, mmap_mode="r", allow_pickle=False)
    splits_map = np.load(paths.splits, mmap_mode="r", allow_pickle=False)
    raw_map = np.load(paths.raw_features, mmap_mode="r", allow_pickle=False)
    row_map = np.load(paths.row_ids, mmap_mode="r", allow_pickle=False)

    task1_train = DiskBackedFlowDataset(
        prepared_dir, preprocessor, partition="train", class_ids=OLD_IDS
    )
    replay_positions = select_disk_replay_rows(
        task1_train, class_ids=OLD_IDS, per_class=5_000, seed=SEED
    )
    replay_x, replay_y, replay_ids = task1_train.batch(replay_positions)
    historical_replay = evidence["large_replay_manifest.json"]
    if not (
        replay_x.shape == (25_000, 54)
        and replay_x.dtype == np.float32
        and np.isfinite(replay_x).all()
        and replay_y.dtype == replay_ids.dtype == np.int64
        and len(np.unique(replay_positions)) == len(np.unique(replay_ids)) == 25_000
        and np.array_equal(replay_ids, np.asarray(historical_replay["original_row_ids"]))
        and np.all(splits_map[replay_positions] == 0)
        and np.array_equal(row_map[replay_positions], replay_ids)
        and np.array_equal(labels_map[replay_positions], replay_y)
        and np.array_equal(preprocessor.transform(raw_map[replay_positions]), replay_x)
        and all(np.count_nonzero(replay_y == class_id) == 5_000 for class_id in OLD_IDS)
        and sha256_array(replay_x) == historical_replay["features_sha256"]
        and sha256_array(replay_y) == historical_replay["labels_sha256"]
        and sha256_array(replay_ids) == historical_replay["original_row_ids_sha256"]
        and historical_replay["preprocessing_sha256"]
        == task1_identity["continual_preprocessing_sha256"]
    ):
        raise ValueError("The 25,000-row replay does not match the saved training-only manifest")
    replay_npz = destination / "verified_large_replay_private.npz"
    replay_npz_sha = _load_or_write_npz(
        replay_npz,
        {"features": replay_x, "labels": replay_y, "original_row_ids": replay_ids},
    )
    replay_record = {
        "status": "reconstructed exactly from the saved retention manifest",
        "source_partition": "fixed training split only",
        "seed": SEED,
        "rows": 25_000,
        "unique_original_rows": 25_000,
        "class_counts": require_supplied_class_buckets(replay_y),
        "features_sha256": sha256_array(replay_x),
        "labels_sha256": sha256_array(replay_y),
        "original_row_ids_sha256": sha256_array(replay_ids),
        "saved_manifest_original_ids_match": True,
        "single_frozen_preprocessing_pass_verified": True,
        "private_npz": replay_npz.name,
        "private_npz_sha256": replay_npz_sha,
    }
    write_or_match(destination / "replay_verification.json", replay_record)

    attack = apply_label_flip(
        replay_y,
        ATTACK_RATE,
        SEED,
        mode="targeted",
        source_class_ids=(1, 2, 3, 4),
        target_class_id=0,
        allowed_class_ids=OLD_IDS,
        original_row_indices=replay_ids,
    )
    if not (
        attack.total_count == 25_000
        and attack.eligible_count == 20_000
        and attack.changed_count == 4_000
        and attack.achieved_eligible_fraction == 0.20
        and attack.achieved_total_fraction == 0.16
        and np.all(replay_y[attack.changed_indices] != 0)
        and np.all(attack.poisoned_labels[attack.changed_indices] == 0)
        and np.array_equal(attack.changed_original_row_indices, replay_ids[attack.changed_indices])
    ):
        raise ValueError("Targeted attack budget or source-to-Benign semantics changed")
    attack_npz = destination / "targeted_attack_audit_private.npz"
    attack_npz_sha = _load_or_write_npz(
        attack_npz,
        {
            "original_labels": replay_y,
            "supplied_labels": attack.poisoned_labels,
            "changed_positions": attack.changed_indices,
            "changed_original_row_ids": attack.changed_original_row_indices,
        },
    )
    attack_record = {
        "mode": attack.mode,
        "seed": attack.seed,
        "requested_rate": attack.requested_rate,
        "budget_denominator": attack.budget_denominator,
        "source_class_ids": list(attack.source_class_ids or ()),
        "target_class_id": attack.target_class_id,
        "total_replay_rows": attack.total_count,
        "eligible_original_attack_rows": attack.eligible_count,
        "changed_rows": attack.changed_count,
        "eligible_fraction": attack.achieved_eligible_fraction,
        "total_replay_fraction": attack.achieved_total_fraction,
        "changed_positions_sha256": sha256_array(attack.changed_indices),
        "changed_original_row_ids_sha256": sha256_array(attack.changed_original_row_indices),
        "supplied_labels_sha256": sha256_array(attack.poisoned_labels),
        "private_audit_npz": attack_npz.name,
        "private_audit_npz_sha256": attack_npz_sha,
        "test_data_used": False,
    }
    write_or_match(destination / "attack_audit.json", attack_record)

    calibration = evidence["prior_calibration"]
    old_validation_for_scope = DiskBackedFlowDataset(
        prepared_dir, preprocessor, partition="validation", class_ids=OLD_IDS
    )
    if len(old_validation_for_scope) != calibration["validation_count"]:
        raise ValueError("Frozen calibration validation scope no longer matches")
    teacher = TabularTransformerClassifier.load(best_path, device="cuda")
    if teacher.class_ids != OLD_IDS:
        raise ValueError("Frozen gate teacher is not the five-class Task 1 model")
    teacher_probabilities = teacher.predict_proba(replay_x)
    gate = apply_label_consistency_gate(
        replay_x,
        attack.poisoned_labels,
        teacher_probabilities,
        teacher.class_ids,
        calibration["threshold"],
    )
    filtered_ids = replay_ids[gate.retained_mask]
    filtered_counts = require_supplied_class_buckets(gate.retained_labels)
    gate_audit = replay_gate_metrics(
        gate.retained_mask,
        attack.changed_indices,
        replay_y,
        attack.poisoned_labels,
        class_ids=OLD_IDS,
    )
    gate_npz = destination / "gate_audit_private.npz"
    gate_npz_sha = _load_or_write_npz(
        gate_npz,
        {
            "scores": gate.scores,
            "retained_mask": gate.retained_mask,
            "original_row_ids": replay_ids,
            "changed_positions": attack.changed_indices,
        },
    )
    gate_record = {
        "gate_inputs": ["frozen replay features", "supplied labels", "Task 1 teacher"],
        "ground_truth_used_for_filtering": False,
        "ground_truth_used_for_post_gate_audit_only": True,
        "teacher_checkpoint_sha256": task1_best_sha,
        "calibration": calibration,
        "calibration_record_sha256": EXPECTED_CALIBRATION_JSON_SHA256,
        "threshold_reused_without_test_tuning": True,
        "supplied_counts_before": require_supplied_class_buckets(attack.poisoned_labels),
        "supplied_counts_after": filtered_counts,
        "gate_metrics": gate_audit,
        "retained_features_sha256": sha256_array(gate.retained_features),
        "retained_labels_sha256": sha256_array(gate.retained_labels),
        "retained_original_row_ids_sha256": sha256_array(filtered_ids),
        "private_audit_npz": gate_npz.name,
        "private_audit_npz_sha256": gate_npz_sha,
    }
    write_or_match(destination / "gate_audit.json", gate_record)
    del teacher, teacher_probabilities
    torch.cuda.empty_cache()

    class_probabilities = fixed_b_probabilities(split_manifest, tuple(BROAD_CLASS_ORDER))
    historical_probabilities = {
        int(key): value
        for key, value in evidence["protocol.json"]["B"]["class_probabilities"].items()
    }
    if class_probabilities != historical_probabilities:
        raise ValueError("Configuration-B class probabilities changed")
    protocol = {
        "source_sha": source_sha,
        "study": "exploratory frozen-B targeted replay poisoning comparison",
        "historical_retention_evidence_sha256": EXPECTED_RETENTION_ZIP_SHA256,
        "prior_task2_evidence_sha256": EXPECTED_PRIOR_TASK2_ZIP_SHA256,
        "prepared_input_sha256": EXPECTED_INPUT_SHA256,
        "task1_best_sha256": task1_best_sha,
        "preprocessing_sha256": task1_identity["continual_preprocessing_sha256"],
        "arms": list(ARMS),
        "shared": {
            "architecture": {key: task1_config["model"][key] for key in
                             ("num_features", "hidden_dim", "num_heads", "num_layers",
                              "mlp_dim", "dropout", "weight_decay")},
            "fresh_identical_seeded_expansion_from_task1_best": True,
            "already_trained_B_checkpoint_used": False,
            "seed": SEED,
            "batch_size": 256,
            "draws_per_epoch": DRAW_COUNT,
            "optimizer_steps_per_epoch": 276,
            "learning_rate": 0.0001,
            "max_epochs": 8,
            "early_stopping_patience": 3,
            "checkpoint_selection": "validation_macro_f1",
            "class_draw_probabilities": {str(key): value
                                         for key, value in class_probabilities.items()},
            "sampling_labels": "supplied replay labels in every arm",
            "preprocessing": "frozen continual Task 1 training-only state; replay not reprocessed",
        },
        "attack": attack_record,
        "gate": {
            "teacher_checkpoint_sha256": task1_best_sha,
            "threshold": calibration["threshold"],
            "fit_scope": calibration["fit_scope"],
            "test_used_for_calibration_or_settings": False,
        },
        "test_policy": (
            "Each arm's validation-macro-F1-selected checkpoint is evaluated once on fixed "
            "test rows only after all three checkpoints and test_open_lock.json are durable."
        ),
        "interpretation": (
            "This tests the frozen configuration-B protocol under replay corruption/filtering; "
            "one seed cannot establish broad robustness."
        ),
    }
    write_or_match(destination / "protocol.json", protocol)
    write_or_match(destination / "environment.json", environment)

    task2_train = DiskBackedFlowDataset(
        prepared_dir, preprocessor, partition="train", class_ids=NEW_IDS
    )
    validation_all = DiskBackedFlowDataset(
        prepared_dir, preprocessor, partition="validation", class_ids=ALL_IDS
    )
    old_validation_ids = old_validation_for_scope.indices
    new_validation_ids = DiskBackedFlowDataset(
        prepared_dir, preprocessor, partition="validation", class_ids=NEW_IDS
    ).indices
    old_validation = DiskBackedFlowDataset(
        prepared_dir, preprocessor, partition="validation", class_ids=ALL_IDS,
        row_ids=old_validation_ids,
    )
    new_validation = DiskBackedFlowDataset(
        prepared_dir, preprocessor, partition="validation", class_ids=ALL_IDS,
        row_ids=new_validation_ids,
    )
    if not (
        len(task2_train) == 70_076
        and len(validation_all) == 1_375_139
        and len(old_validation) == 1_360_123
        and len(new_validation) == 15_016
    ):
        raise ValueError("Task 2 train or clean validation split count changed")

    arm_inputs = {
        "clean_B": (replay_x, replay_y, replay_ids),
        "targeted20_B": (replay_x, attack.poisoned_labels, replay_ids),
        "targeted20_filtered_B": (
            gate.retained_features, gate.retained_labels, filtered_ids
        ),
    }
    class_names = dict(enumerate(BROAD_CLASS_ORDER))
    arm_summaries: dict[str, dict] = {}
    expanded_initial_hash: str | None = None
    for name in ARMS:
        folder = destination / name
        folder.mkdir(exist_ok=True)
        replay_features, supplied_labels, original_ids = arm_inputs[name]
        supplied_counts = require_supplied_class_buckets(supplied_labels)
        train_view = Task2ReplayView(
            task2_train,
            replay_features,
            supplied_labels,
            original_ids,
            class_ids=ALL_IDS,
            draws_per_epoch=DRAW_COUNT,
            class_draw_probabilities=class_probabilities,
        )
        planned = train_view.planned_class_draw_counts
        if not (
            sum(planned[class_id] for class_id in OLD_IDS) == 44_110
            and all(planned[class_id] == 8_822 for class_id in NEW_IDS)
        ):
            raise ValueError(f"{name}: frozen configuration-B integer quotas changed")

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        model = TabularTransformerClassifier.load(best_path, device="cuda")
        if model.class_ids != OLD_IDS or model.config.learning_rate != 0.001:
            raise ValueError("Task 1 mapping or inherited learning rate changed")
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        model.add_classes(NEW_IDS)
        initial_hash = state_hash(model)
        if expanded_initial_hash is None:
            expanded_initial_hash = initial_hash
        if initial_hash != expanded_initial_hash or initial_hash != HISTORICAL_B_INITIAL_SHA256:
            raise ValueError(f"{name}: expanded initial weights differ from historical B")
        model.config = replace(
            model.config,
            learning_rate=0.0001,
            batch_size=256,
            epochs=8,
            early_stopping_patience=3,
            training_sampler="class_balanced",
            sampler_seed=SEED,
            mixed_precision=False,
        )
        if not (
            model.class_ids == ALL_IDS
            and model.config.num_classes == 8
            and (model.config.hidden_dim, model.config.num_heads, model.config.num_layers,
                 model.config.mlp_dim, model.config.batch_size, model.config.seed)
            == (64, 4, 2, 128, 256, SEED)
        ):
            raise ValueError("Expanded model architecture differs from frozen B")

        identity = {
            "source_sha": source_sha,
            "task1_best_sha256": task1_best_sha,
            "task1_data_identity": task1_identity,
            "condition": name,
            "task2_training_rows": len(task2_train),
            "validation_rows": len(validation_all),
            "draws_per_epoch": DRAW_COUNT,
            "replay_features_sha256": sha256_array(replay_features),
            "supplied_replay_labels_sha256": sha256_array(supplied_labels),
            "replay_original_ids_sha256": sha256_array(original_ids),
            "stored_supplied_counts": supplied_counts,
            "class_draw_probabilities": {str(key): value
                                         for key, value in class_probabilities.items()},
            "expanded_initial_sha256": initial_hash,
            "checkpoint_selection": "validation_macro_f1",
        }
        config = {
            "condition": name,
            "source_sha": source_sha,
            "model": asdict(model.config),
            "data_identity": identity,
            "stored_replay_rows": len(supplied_labels),
            "stored_supplied_counts": supplied_counts,
            "present_classes": list(train_view.present_class_ids),
            "planned_class_draw_counts": {str(key): value for key, value in planned.items()},
        }
        write_or_match(folder / "arm_config.json", config)
        latest_path = folder / "latest.pt"
        if not latest_path.exists() and any(
            (folder / filename).exists() for filename in
            ("best.pt", "history.json", "training_manifest.json", "validation_metrics.json")
        ):
            raise RuntimeError(f"{name}: partial arm folder lacks authoritative latest.pt")
        result = train_full_disk_backed(
            model,
            train_view,
            validation_all,
            output_dir=folder,
            data_identity=identity,
            resume=latest_path.exists(),
            max_epochs_this_call=args.max_epochs_this_call,
            progress_every_batches=100,
            checkpoint_selection="validation_macro_f1",
            report=lambda message: print(message, flush=True),
        )
        if result["paused"]:
            pause = {
                "condition": name,
                "completed_epoch": result["completed_epoch"],
                "resume_run_dir": str(destination),
                "message": "Preserve the complete folder and pass it with --resume-run-dir.",
            }
            atomic_json(destination / "PAUSED.json", pause)
            raise RuntimeError(
                f"{name} paused at epoch {result['completed_epoch']}; preserve {destination} "
                "and resume from the durable latest checkpoint"
            )
        history = result["history"]
        if not (
            history["training_rows_processed"] == DRAW_COUNT * result["completed_epoch"]
            and history["optimizer_steps"] == 276 * result["completed_epoch"]
            and history["checkpoint_selection"] == "validation_macro_f1"
        ):
            raise ValueError(f"{name}: actual work counters differ from completed epochs")
        for row in history["epochs"]:
            exposure = row["sampling_exposure"]
            if not (
                row["train_rows"] == exposure["draws"] == DRAW_COUNT
                and exposure["sampled_counts"] == exposure["planned_class_draw_counts"]
                and exposure["stored_replay_rows"] == len(supplied_labels)
                and exposure["sampler_seed"] == SEED
            ):
                raise ValueError(f"{name}: recorded sampler exposure differs from frozen B")

        latest = torch.load(latest_path, map_location="cpu", weights_only=False)
        selected_model = TabularTransformerClassifier.load(folder / "best.pt", device="cuda")
        if not (
            selected_model.class_ids == ALL_IDS
            and latest["best_epoch"] == result["best_epoch"]
            and latest["checkpoint_selection"] == "validation_macro_f1"
            and all(torch.equal(tensor.detach().cpu(), latest["best_state_dict"][key])
                    for key, tensor in selected_model.network.state_dict().items())
        ):
            raise ValueError(f"{name}: best/latest checkpoint consistency failed")
        check_x, _, _ = validation_all.batch(validation_all.indices[:256])
        reloaded = TabularTransformerClassifier.load(folder / "best.pt", device="cuda")
        if not np.array_equal(selected_model.predict(check_x), reloaded.predict(check_x)):
            raise ValueError(f"{name}: bounded reload predictions differ")
        checkpoint_sha = sha256_file(folder / "best.pt")
        write_or_match(folder / "reload_verification.json", {
            "best_epoch": result["best_epoch"],
            "latest_best_state_equal": True,
            "bounded_validation_predictions_identical": True,
            "best_checkpoint_sha256": checkpoint_sha,
            "checkpoint_selection": "validation_macro_f1",
        })
        validation_path = folder / "validation_metrics.json"
        if validation_path.exists():
            validation_record = json.loads(validation_path.read_text())
            if validation_record["best_checkpoint_sha256"] != checkpoint_sha:
                raise ValueError(f"{name}: saved validation metrics are stale")
        else:
            validation_record = {
                "condition": name,
                "source_sha": source_sha,
                "best_checkpoint_sha256": checkpoint_sha,
                "selection_partition": "fixed clean validation only",
                "metrics": _evaluate_validation(
                    selected_model, old_validation, new_validation, class_names
                ),
            }
            atomic_json(validation_path, validation_record)
        arm_summaries[name] = {
            "completed_epoch": result["completed_epoch"],
            "best_epoch": result["best_epoch"],
            "early_stopped": result["early_stopped"],
            "best_validation_macro_f1": result["best_selection_value"],
            "actual_training_draws": history["training_rows_processed"],
            "actual_optimizer_steps": history["optimizer_steps"],
            "actual_validation_rows_processed": history["validation_rows_processed"],
            "train_seconds": sum(row["train_seconds"] for row in history["epochs"]),
            "validation_seconds": sum(row["validation_seconds"] for row in history["epochs"]),
            "stored_replay_rows": len(supplied_labels),
            "stored_supplied_counts": supplied_counts,
            "expanded_initial_sha256": initial_hash,
            "best_checkpoint_sha256": checkpoint_sha,
            "exposure_by_epoch": [row["sampling_exposure"] for row in history["epochs"]],
        }
        write_or_match(folder / "arm_summary.json", arm_summaries[name])
        save_training_curves(history, folder)
        print(
            name,
            "actual epochs/steps:",
            result["completed_epoch"],
            history["optimizer_steps"],
            "best epoch:",
            result["best_epoch"],
        )

    if len({record["expanded_initial_sha256"] for record in arm_summaries.values()}) != 1:
        raise ValueError("The three arms did not share identical expanded initial weights")
    test_open_lock = {
        "source_sha": source_sha,
        "status": "all three validation-selected checkpoints durable before test evaluation",
        "test_used_for_attack_gate_or_training_settings": False,
        "arms": {
            name: {
                "best_epoch": arm_summaries[name]["best_epoch"],
                "completed_epoch": arm_summaries[name]["completed_epoch"],
                "best_checkpoint_sha256": arm_summaries[name]["best_checkpoint_sha256"],
                "validation_metrics_sha256": sha256_file(
                    destination / name / "validation_metrics.json"
                ),
            }
            for name in ARMS
        },
    }
    write_or_match(destination / "test_open_lock.json", test_open_lock)

    task1_reference = json.loads((task1_dir / "task1_test_metrics.json").read_text())
    if not (
        task1_reference["data_identity"] == task1_identity
        and task1_reference["best_checkpoint_sha256"] == task1_best_sha
        and task1_reference["metrics"]["class_ids"] == list(OLD_IDS)
    ):
        raise ValueError("Task 1-before test reference is incompatible")
    old_test_ids = DiskBackedFlowDataset(
        prepared_dir, preprocessor, partition="test", class_ids=OLD_IDS
    ).indices
    new_test_ids = DiskBackedFlowDataset(
        prepared_dir, preprocessor, partition="test", class_ids=NEW_IDS
    ).indices
    old_test = DiskBackedFlowDataset(
        prepared_dir, preprocessor, partition="test", class_ids=ALL_IDS,
        row_ids=old_test_ids,
    )
    new_test = DiskBackedFlowDataset(
        prepared_dir, preprocessor, partition="test", class_ids=ALL_IDS,
        row_ids=new_test_ids,
    )
    if len(old_test) != 1_360_119 or len(new_test) != 15_015:
        raise ValueError("Fixed test split counts changed")

    final_results: dict[str, dict] = {}
    for name in ARMS:
        folder = destination / name
        selected_model = TabularTransformerClassifier.load(folder / "best.pt", device="cuda")
        partition_metrics: dict[str, dict] = {}
        for label, view in (("old", old_test), ("new", new_test)):
            path = folder / f"{label}_test_metrics.json"
            if path.exists():
                record = json.loads(path.read_text())
                if record["best_checkpoint_sha256"] != arm_summaries[name]["best_checkpoint_sha256"]:
                    raise ValueError(f"{name}: saved {label} test result is stale")
            else:
                torch.cuda.synchronize()
                started = perf_counter()
                metrics = evaluate_disk_backed(
                    selected_model,
                    view,
                    batch_size=256,
                    class_names=class_names,
                    benign_class_id=0,
                )
                torch.cuda.synchronize()
                record = {
                    "condition": name,
                    "source_sha": source_sha,
                    "best_checkpoint_sha256": arm_summaries[name]["best_checkpoint_sha256"],
                    "partition": f"fixed {label}-class test rows",
                    "inference_seconds": perf_counter() - started,
                    "metrics": metrics,
                }
                atomic_json(path, record)
            partition_metrics[label] = record["metrics"]
        summary = summarize_task2_counts(
            task1_reference["metrics"],
            np.asarray(partition_metrics["old"]["confusion_matrix"], dtype=np.int64),
            np.asarray(partition_metrics["new"]["confusion_matrix"], dtype=np.int64),
            class_names=class_names,
        )
        combined_record = {
            "condition": name,
            "source_sha": source_sha,
            "best_checkpoint_sha256": arm_summaries[name]["best_checkpoint_sha256"],
            "test_open_lock_sha256": sha256_file(destination / "test_open_lock.json"),
            "test_used_for_settings": False,
            "metrics": summary["combined_test"],
        }
        write_or_match(folder / "combined_test_metrics.json", combined_record)
        per_class = {
            str(class_id): {
                "name": BROAD_CLASS_ORDER[class_id],
                "recall": summary["combined_test"]["per_class"][str(class_id)]["recall"],
                "f1": summary["combined_test"]["per_class"][str(class_id)]["f1"],
                "support": summary["combined_test"]["per_class"][str(class_id)]["support"],
            }
            for class_id in ALL_IDS
        }
        final = {
            "condition": name,
            "actual_training": arm_summaries[name],
            "task1_before": task1_reference["metrics"],
            "old_test": summary["old_test"],
            "new_test": summary["new_test"],
            "combined_test": summary["combined_test"],
            "old_focused_macro_f1": summary["old_focused_macro_f1"],
            "new_focused_macro_f1": summary["new_focused_macro_f1"],
            "old_focused_balanced_accuracy": summary["old_focused_balanced_accuracy"],
            "new_focused_balanced_accuracy": summary["new_focused_balanced_accuracy"],
            "benign_fpr": summary["benign_fpr"],
            "old_attack_to_benign": summary["old_attack_to_benign"],
            "forgetting": summary["forgetting"],
            "per_class_recall_f1": per_class,
            "gate_audit": gate_audit if name == "targeted20_filtered_B" else None,
        }
        write_or_match(folder / "test_summary.json", final)
        expected_plots = tuple(
            folder / f"combined_test_{suffix}.png" for suffix in
            ("confusion_counts", "confusion_row_normalized", "per_class_recall_f1")
        )
        existing = [path.exists() for path in expected_plots]
        if any(existing) and not all(existing):
            raise RuntimeError(f"{name}: incomplete combined-test plot set")
        if not any(existing):
            save_result_plots(
                folder / "combined_test_metrics.json", folder, prefix="combined_test"
            )
        final_results[name] = final

    historical = evidence["selected_test_summary.json"]
    historical_validation = evidence["B_large_proportional_lr1e4/validation_metrics.json"]
    clean = final_results["clean_B"]
    clean_validation = json.loads(
        (destination / "clean_B" / "validation_metrics.json").read_text()
    )
    reproducibility_deltas = {
        "test_combined_accuracy": _delta(
            clean["combined_test"]["accuracy"], historical["combined_test"]["accuracy"]
        ),
        "test_combined_macro_f1": _delta(
            clean["combined_test"]["macro_f1"], historical["combined_test"]["macro_f1"]
        ),
        "test_benign_fpr": _delta(clean["benign_fpr"], historical["benign_fpr"]),
        "test_old_accuracy": _delta(clean["old_test"]["accuracy"], historical["old_test"]["accuracy"]),
        "test_new_accuracy": _delta(clean["new_test"]["accuracy"], historical["new_test"]["accuracy"]),
        "validation_combined_macro_f1": _delta(
            clean_validation["metrics"]["combined_validation"]["macro_f1"],
            historical_validation["metrics"]["combined_validation"]["macro_f1"],
        ),
    }
    exact_metrics = all(abs(value) <= 1e-15 for value in reproducibility_deltas.values())
    clean_reproducibility = {
        "historical_source_sha": HISTORICAL_RETENTION_SOURCE,
        "current_source_sha": source_sha,
        "historical_B_checkpoint_sha256": historical["selected_checkpoint_sha256"],
        "current_clean_checkpoint_sha256": arm_summaries["clean_B"]["best_checkpoint_sha256"],
        "same_replay_hashes": (
            replay_record["features_sha256"]
            == evidence["large_replay_manifest.json"]["features_sha256"]
            and replay_record["labels_sha256"]
            == evidence["large_replay_manifest.json"]["labels_sha256"]
            and replay_record["original_row_ids_sha256"]
            == evidence["large_replay_manifest.json"]["original_row_ids_sha256"]
        ),
        "same_expanded_initial_weights": (
            arm_summaries["clean_B"]["expanded_initial_sha256"]
            == evidence["B_large_proportional_lr1e4/arm_summary.json"][
                "expanded_initial_sha256"
            ]
        ),
        "historical_environment": evidence["environment.json"],
        "current_environment": environment,
        "metric_deltas_current_minus_historical": reproducibility_deltas,
        "metrics_exact_within_1e_15": exact_metrics,
        "interpretation": (
            "The clean arm reproduced the uploaded B metrics exactly within 1e-15."
            if exact_metrics else
            "The clean arm did not reproduce every uploaded B metric exactly. Protocol, replay, "
            "and initial-weight hashes were checked; the discrepancy is reported without tuning. "
            "Runtime/library or GPU nondeterminism is a hypothesis, not an established cause."
        ),
    }
    write_or_match(destination / "clean_reproducibility.json", clean_reproducibility)

    comparison = {
        "source_sha": source_sha,
        "study": "frozen configuration-B targeted replay poisoning comparison",
        "test_used_for_settings": False,
        "attack_budget": attack_record,
        "gate_tradeoff": gate_audit,
        "clean_reproducibility": clean_reproducibility,
        "arms": {
            name: {
                "completed_epoch": final_results[name]["actual_training"]["completed_epoch"],
                "best_epoch": final_results[name]["actual_training"]["best_epoch"],
                "optimizer_steps": final_results[name]["actual_training"]["actual_optimizer_steps"],
                "combined_accuracy": final_results[name]["combined_test"]["accuracy"],
                "combined_macro_f1": final_results[name]["combined_test"]["macro_f1"],
                "old_accuracy": final_results[name]["old_test"]["accuracy"],
                "new_accuracy": final_results[name]["new_test"]["accuracy"],
                "benign_fpr": final_results[name]["benign_fpr"],
                "old_attack_to_benign": final_results[name]["old_attack_to_benign"],
                "forgetting": final_results[name]["forgetting"],
                "per_class_recall_f1": final_results[name]["per_class_recall_f1"],
            }
            for name in ARMS
        },
        "limitations": [
            "This is one fixed-seed exploratory follow-up; it cannot establish broad robustness.",
            "The frozen B control previously improved old-class retention but acquired new classes weakly.",
            "Ground-truth poison masks were used only for post-gate audit metrics.",
            "Filtering can remove clean replay and can leave harmful poisoned rows; both counts are reported.",
            "Aggregate accuracy must be read with new-class and per-class metrics.",
        ],
    }
    write_or_match(destination / "comparison.json", comparison)
    manifest = {
        "status": "completed frozen-B clean/targeted/filtered comparison",
        "source_sha": source_sha,
        "environment": environment,
        "historical_artifacts_modified": False,
        "verified_replay_npz": replay_npz.name,
        "verified_replay_npz_sha256": replay_npz_sha,
        "test_open_lock_sha256": sha256_file(destination / "test_open_lock.json"),
        "comparison_sha256": sha256_file(destination / "comparison.json"),
        "actual_epochs_and_steps": {
            name: {
                "epochs": arm_summaries[name]["completed_epoch"],
                "best_epoch": arm_summaries[name]["best_epoch"],
                "steps": arm_summaries[name]["actual_optimizer_steps"],
            }
            for name in ARMS
        },
    }
    write_or_match(destination / "study_manifest.json", manifest)
    print(json.dumps(comparison["arms"], indent=2))
    print("Gate poison/clean rejection counts:", json.dumps(gate_audit, indent=2))
    print("Preserve the complete private output folder:", destination)


if __name__ == "__main__":
    main()
