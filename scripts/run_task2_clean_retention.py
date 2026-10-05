#!/usr/bin/env python3
"""Run the bounded clean Task 2 retention study on a Kaggle GPU.

This entry point intentionally reuses a completed Task 1 checkpoint and the
frozen continual preprocessor. It never trains Task 1, mutates either input,
or evaluates an unselected Task 2 configuration on the test partition.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
from time import perf_counter
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


EXPECTED_PREP_SOURCE_SHA = "5c0d519596bc6c4b99948126443635c2f7eb15f1"
EXPECTED_TASK1_SOURCE_SHA = "e6ebfcf7a9beec4d2b8fd5245e7c62314b9fadf9"
EXPECTED_INPUT_SHA256 = "666af53c788c79312b421c607289cb555c2335ae51ba39b2700a357595fa8dba"
EXPECTED_TASK1_BEST_SHA256 = "aed830e152d61525c641a95ac8eded661caa8c318d9aed910abb954782454de6"
OLD_IDS = tuple(range(5))
NEW_IDS = (5, 6, 7)
ALL_IDS = tuple(range(8))
CONDITIONS = ("A_small_uniform_lr1e4", "B_large_proportional_lr1e4")
DRAW_COUNT = 70_576
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bounded clean Task 2 retention study; CUDA/Kaggle only."
    )
    parser.add_argument("--prepared-dir", required=True, type=Path)
    parser.add_argument("--task1-run-dir", required=True, type=Path)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output-dir", type=Path)
    destination.add_argument("--resume-run-dir", type=Path)
    parser.add_argument("--max-epochs-this-call", type=int)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_or_match(path: Path, value: dict) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Existing {path} belongs to a different study state")
    else:
        atomic_json(path, value)


def require_kaggle_path(path: Path, label: str) -> Path:
    resolved = path.resolve()
    roots = (Path("/kaggle/input"), Path("/kaggle/working"))
    if not resolved.is_dir() or not any(resolved.is_relative_to(root) for root in roots):
        raise ValueError(f"{label} must be an existing /kaggle/input or /kaggle/working folder")
    return resolved


def output_folder(args: argparse.Namespace) -> Path:
    if args.resume_run_dir is not None:
        previous = require_kaggle_path(args.resume_run_dir, "resume run")
        if previous.is_relative_to(Path("/kaggle/working")):
            return previous
        destination = Path("/kaggle/working") / (
            "task2_clean_retention_resume_"
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
        "task2_clean_retention_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    )
    destination.mkdir(exist_ok=False)
    return destination


def state_hash(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.network.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def focused_metrics(metrics: dict, class_ids: tuple[int, ...]) -> dict:
    return {
        "accuracy": metrics["accuracy"],
        "macro_f1": float(np.mean([
            metrics["per_class"][str(class_id)]["f1"] for class_id in class_ids
        ])),
        "balanced_accuracy": float(np.mean([
            metrics["per_class"][str(class_id)]["recall"] for class_id in class_ids
        ])),
        "rows": metrics["total_rows"],
    }


def combined_partition_metrics(old_metrics: dict, new_metrics: dict,
                               class_names: dict[int, str]) -> dict:
    from src.evaluation.streaming import ConfusionAccumulator

    old_counts = np.asarray(old_metrics["confusion_matrix"], dtype=np.int64)
    new_counts = np.asarray(new_metrics["confusion_matrix"], dtype=np.int64)
    accumulator = ConfusionAccumulator(ALL_IDS)
    accumulator.counts = old_counts + new_counts
    combined = accumulator.metrics(class_names=class_names, benign_class_id=0)
    return {
        "old_validation": old_metrics,
        "new_validation": new_metrics,
        "combined_validation": combined,
        "old_focused": focused_metrics(old_metrics, OLD_IDS),
        "new_focused": focused_metrics(new_metrics, NEW_IDS),
        "benign_fpr": combined["benign_false_positive"]["rate"],
    }


def save_training_curves(history: dict, destination: Path) -> None:
    import matplotlib.pyplot as plt

    target = destination / "training_curves.png"
    if target.exists():
        return
    rows = history["epochs"]
    epochs = [row["epoch"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in rows], marker="o", label="train")
    axes[0].plot(epochs, [row["validation_loss"] for row in rows], marker="o",
                 label="clean validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy", title="Loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["validation_macro_f1"] for row in rows], marker="o",
                 label="macro-F1")
    axes[1].plot(epochs, [row["validation_accuracy"] for row in rows], marker="o",
                 label="accuracy")
    axes[1].set(xlabel="Epoch", ylabel="Score", ylim=(0, 1.05),
                title="Full clean validation")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(target, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.max_epochs_this_call is not None and args.max_epochs_this_call < 1:
        raise ValueError("max-epochs-this-call must be positive")

    import torch
    from src.continual_learning.disk_replay import select_disk_replay_rows
    from src.data import BROAD_CLASS_ORDER, FittedPreprocessor
    from src.data.large_data import large_data_paths, verify_full_partitions
    from src.evaluation.plots import save_result_plots
    from src.evaluation.streaming import evaluate_disk_backed, summarize_task2_counts
    from src.models import TabularTransformerClassifier
    from src.training.disk_backed import DiskBackedFlowDataset
    from src.training.full_run import train_full_disk_backed
    from src.training.task2_replay import Task2ReplayView

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; enable a Kaggle GPU and restart the session")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    prepared_dir = require_kaggle_path(args.prepared_dir, "prepared input")
    task1_dir = require_kaggle_path(args.task1_run_dir, "Task 1 run")
    destination = output_folder(args)
    if (destination == prepared_dir or destination == task1_dir or
            destination.is_relative_to(prepared_dir) or
            destination.is_relative_to(task1_dir)):
        raise ValueError("Output must be separate from both read-only inputs")
    if shutil.disk_usage("/kaggle/working").free < 750_000_000:
        raise RuntimeError("Less than 750 MB is free under /kaggle/working")

    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    environment = {
        "python": platform.python_version(), "numpy": np.__version__,
        "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_device_count": torch.cuda.device_count(),
    }

    prep_manifest = json.loads((prepared_dir / "manifest.json").read_text())
    split_manifest = json.loads((prepared_dir / "split_manifest.json").read_text())
    features_manifest = json.loads((prepared_dir / "features_manifest.json").read_text())
    continual_record = json.loads(
        (prepared_dir / "preprocessing_continual.json").read_text()
    )
    task1_final = json.loads((task1_dir / "final_manifest.json").read_text())
    task1_config = json.loads((task1_dir / "run_config.json").read_text())
    task1_training = json.loads((task1_dir / "training_manifest.json").read_text())
    task1_reload = json.loads((task1_dir / "reload_verification.json").read_text())
    replay_manifest = json.loads((task1_dir / "replay_manifest.json").read_text())
    paths = large_data_paths(prepared_dir)

    if not (prep_manifest["completed_preparation_and_benchmark_only"] is True and
            prep_manifest["source_sha"] == EXPECTED_PREP_SOURCE_SHA and
            prep_manifest["input_sha256"] == EXPECTED_INPUT_SHA256 and
            prep_manifest["input_rows"] == split_manifest["row_count"] ==
            features_manifest["row_count"] == 9_167_581):
        raise ValueError("Prepared-data manifest does not match the verified fixed input")
    if not (split_manifest["seed"] == SEED and
            split_manifest["class_order"] == list(BROAD_CLASS_ORDER) and
            split_manifest["split_codes"] == {"train": 0, "validation": 1, "test": 2}):
        raise ValueError("Prepared split definition changed")
    if not (features_manifest["dtype"] == "float32" and
            features_manifest["targets_excluded"] == ["Label", "ClassLabel"] and
            len(features_manifest["feature_names"]) == 54):
        raise ValueError("Prepared feature contract changed")
    if not (continual_record["scope"] == "continual" and
            continual_record["fit_rows"] == 6_347_232 and
            continual_record["feature_names"] == features_manifest["feature_names"] and
            continual_record["reservoir_seed"] == SEED and
            continual_record["reservoir_size_used"] == 200_000):
        raise ValueError("Frozen continual preprocessing does not match Task 1")
    if verify_full_partitions(paths, 9_167_581)["counts"] != split_manifest["counts"]:
        raise ValueError("Prepared arrays do not match the fixed split manifest")

    task1_identity = task1_final["data_identity"]
    if not (task1_final["status"] == "completed full-data Task 1 foundation only" and
            task1_final["task2_trained"] is False and
            task1_final["source_sha"] == task1_config["source_sha"] ==
            EXPECTED_TASK1_SOURCE_SHA and
            task1_final["class_ids"] == task1_config["class_ids"] ==
            task1_training["class_ids"] == list(OLD_IDS) and
            task1_final["completed_epoch"] == 12 and task1_final["best_epoch"] == 9 and
            task1_final["data_identity"] == task1_config["data_identity"] ==
            task1_training["data_identity"]):
        raise ValueError("Task 1 run is not the verified completed foundation")
    if not (task1_identity["input_sha256"] == EXPECTED_INPUT_SHA256 and
            task1_identity["prep_source_sha"] == EXPECTED_PREP_SOURCE_SHA and
            task1_identity["preprocessing_scope"] ==
            "Task 1 training global IDs 0-4 only" and
            task1_identity["task1_class_ids"] == list(OLD_IDS)):
        raise ValueError("Task 1 data/preprocessing identity changed")
    identity_files = (
        (paths.raw_features, "raw_features_sha256"),
        (paths.labels, "labels_sha256"),
        (paths.splits, "splits_sha256"),
        (paths.row_ids, "row_ids_sha256"),
        (prepared_dir / "split_manifest.json", "split_manifest_sha256"),
        (prepared_dir / "preprocessing_continual.json", "continual_preprocessing_sha256"),
    )
    for path, key in identity_files:
        if sha256_file(path) != task1_identity[key]:
            raise ValueError(f"Prepared identity mismatch: {key}")
    if sha256_file(task1_dir / "preprocessing_continual.json") != \
            task1_identity["continual_preprocessing_sha256"]:
        raise ValueError("Task 1 copy of the continual preprocessor changed")

    best_path = task1_dir / "best.pt"
    task1_best_sha = sha256_file(best_path)
    if not (task1_best_sha == EXPECTED_TASK1_BEST_SHA256 ==
            task1_final["best_checkpoint_sha256"] ==
            task1_reload["best_checkpoint_sha256"]):
        raise ValueError("Task 1 best checkpoint hash changed")
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    if not (checkpoint["class_ids"] == list(OLD_IDS) and
            checkpoint["data_identity"] == task1_identity and
            checkpoint["best_epoch"] == task1_final["best_epoch"]):
        raise ValueError("Task 1 checkpoint metadata is incompatible")

    preprocessor = FittedPreprocessor(
        tuple(continual_record["feature_names"]),
        np.asarray(continual_record["imputation_values"], dtype=np.float64),
        np.asarray(continual_record["means"], dtype=np.float64),
        np.asarray(continual_record["scales"], dtype=np.float64),
    )
    with np.load(task1_dir / "replay_buffer.npz", allow_pickle=False) as saved:
        replay_x = saved["features"].copy()
        replay_y = saved["labels"].copy()
        replay_ids = saved["original_row_ids"].copy()
    if not (replay_x.shape == (500, 54) and replay_x.dtype == np.float32 and
            np.isfinite(replay_x).all() and replay_y.shape == replay_ids.shape == (500,) and
            replay_y.dtype == replay_ids.dtype == np.int64 and
            len(np.unique(replay_ids)) == 500 and
            np.array_equal(replay_ids, np.asarray(replay_manifest["original_row_ids"])) and
            replay_manifest["seed"] == SEED and
            replay_manifest["rows_per_class"] == 100 and
            replay_manifest["preprocessing_sha256"] ==
            task1_identity["continual_preprocessing_sha256"] and
            all(np.count_nonzero(replay_y == class_id) == 100 for class_id in OLD_IDS) and
            replay_manifest["source_partition"] == "original training only"):
        raise ValueError("Existing 500-row replay buffer is incompatible")

    labels_map = np.load(paths.labels, mmap_mode="r", allow_pickle=False)
    splits_map = np.load(paths.splits, mmap_mode="r", allow_pickle=False)
    raw_map = np.load(paths.raw_features, mmap_mode="r", allow_pickle=False)
    row_map = np.load(paths.row_ids, mmap_mode="r", allow_pickle=False)
    if not (np.all(splits_map[replay_ids] == 0) and
            np.array_equal(row_map[replay_ids], replay_ids) and
            np.array_equal(labels_map[replay_ids], replay_y) and
            np.array_equal(preprocessor.transform(raw_map[replay_ids]), replay_x)):
        raise ValueError("Existing replay provenance or frozen transformation changed")

    task1_train = DiskBackedFlowDataset(
        prepared_dir, preprocessor, partition="train", class_ids=OLD_IDS
    )
    large_positions = select_disk_replay_rows(
        task1_train, class_ids=OLD_IDS, per_class=5_000, seed=SEED
    )
    large_x, large_y, large_ids = task1_train.batch(large_positions)
    if not (large_x.shape == (25_000, 54) and large_x.dtype == np.float32 and
            np.isfinite(large_x).all() and len(np.unique(large_positions)) == 25_000 and
            len(np.unique(large_ids)) == 25_000 and
            np.all(splits_map[large_positions] == 0) and
            np.array_equal(row_map[large_positions], large_ids) and
            all(np.count_nonzero(large_y == class_id) == 5_000 for class_id in OLD_IDS)):
        raise ValueError("Large replay selection is not distinct, balanced, and training-only")
    # DiskBackedFlowDataset.batch applies the frozen transform exactly once.
    if not np.array_equal(preprocessor.transform(raw_map[large_positions]), large_x):
        raise ValueError("Large replay does not match one frozen preprocessing pass")

    old_training_counts = {
        class_id: int(split_manifest["counts"][BROAD_CLASS_ORDER[class_id]]["train"])
        for class_id in OLD_IDS
    }
    old_training_total = sum(old_training_counts.values())
    class_probabilities_b = {
        class_id: 0.625 * old_training_counts[class_id] / old_training_total
        for class_id in OLD_IDS
    }
    class_probabilities_b.update({class_id: 0.125 for class_id in NEW_IDS})
    if not np.isclose(sum(class_probabilities_b.values()), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("Configuration B class probabilities do not sum to one")

    hypotheses = {
        "status": "predeclared hypotheses, not established causes",
        "uniform_eight_class_benign_expected_exposure": 0.125,
        "existing_replay_distinct_rows_per_old_class": 100,
        "historical_task2_inherited_learning_rate": 0.001,
        "historical_test_results_already_inspected": True,
        "study_label": "exploratory follow-up",
        "configuration_b_interpretation": (
            "combined replay-size and sampling intervention; it does not isolate either effect"
        ),
    }
    selection_rule = {
        "checkpoint": "maximize full clean eight-class validation macro-F1",
        "configuration": "maximize selected-checkpoint clean validation macro-F1",
        "exact_ties": "lower validation Benign FPR, then higher old focused macro-F1, then name",
        "test_used_for_selection": False,
    }
    protocol = {
        "source_sha": source_sha,
        "prepared_input_sha256": EXPECTED_INPUT_SHA256,
        "task1_source_sha": EXPECTED_TASK1_SOURCE_SHA,
        "task1_best_sha256": task1_best_sha,
        "task1_data_identity": task1_identity,
        "preprocessing_sha256": task1_identity["continual_preprocessing_sha256"],
        "conditions": list(CONDITIONS), "hypotheses": hypotheses,
        "selection_rule": selection_rule,
        "shared": {
            "architecture": {key: task1_config["model"][key] for key in
                             ("num_features", "hidden_dim", "num_heads", "num_layers",
                              "mlp_dim", "dropout", "weight_decay")},
            "expanded_from_same_task1_best": True, "seed": SEED,
            "batch_size": 256, "draws_per_epoch": DRAW_COUNT,
            "optimizer_steps_per_epoch": 276, "learning_rate": 0.0001,
            "max_epochs": 8, "early_stopping_patience": 3,
            "checkpoint_selection": "validation_macro_f1",
            "preprocessing": "frozen continual Task 1 training-only state",
        },
        "A": {"stored_replay_rows": 500, "distinct_per_old_class": 100,
              "sampler": "existing uniform-over-eight-classes stochastic sampler"},
        "B": {"stored_replay_rows": 25_000, "distinct_per_old_class": 5_000,
              "old_draw_share": 0.625, "old_distribution": "original old training proportions",
              "new_draw_share": 0.375, "new_distribution": "uniform",
              "class_probabilities": {str(k): v for k, v in class_probabilities_b.items()}},
    }
    write_or_match(destination / "protocol.json", protocol)
    write_or_match(destination / "environment.json", environment)
    large_replay_manifest = {
        "seed": SEED, "source_partition": "fixed training split only",
        "rows_per_old_class": 5_000, "total_rows": 25_000,
        "class_counts": {str(class_id): int(np.count_nonzero(large_y == class_id))
                         for class_id in OLD_IDS},
        "original_row_ids": large_ids.tolist(),
        "features_sha256": sha256_array(large_x),
        "labels_sha256": sha256_array(large_y),
        "original_row_ids_sha256": sha256_array(large_ids),
        "preprocessing_sha256": task1_identity["continual_preprocessing_sha256"],
    }
    write_or_match(destination / "large_replay_manifest.json", large_replay_manifest)
    print("Verified immutable inputs, fixed splits, Task 1 checkpoint, and replay isolation")
    print("Hypotheses (not causes):", hypotheses)
    print("Selection rule locked before training/test:", selection_rule)

    task2_train = DiskBackedFlowDataset(
        prepared_dir, preprocessor, partition="train", class_ids=NEW_IDS
    )
    validation_all = DiskBackedFlowDataset(
        prepared_dir, preprocessor, partition="validation", class_ids=ALL_IDS
    )
    old_validation_ids = DiskBackedFlowDataset(
        prepared_dir, preprocessor, partition="validation", class_ids=OLD_IDS
    ).indices
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
    if not (len(task2_train) == 70_076 and len(validation_all) == 1_375_139 and
            len(old_validation) == 1_360_123 and len(new_validation) == 15_016):
        raise ValueError("Task 2 train or clean validation split count changed")

    arm_inputs = {
        CONDITIONS[0]: (replay_x, replay_y, replay_ids, None),
        CONDITIONS[1]: (large_x, large_y, large_ids, class_probabilities_b),
    }
    expanded_initial_hash = None
    arm_summaries = {}
    class_names = dict(enumerate(BROAD_CLASS_ORDER))
    for name in CONDITIONS:
        folder = destination / name
        folder.mkdir(exist_ok=True)
        replay_features, replay_labels, replay_original_ids, probabilities = arm_inputs[name]
        train_view = Task2ReplayView(
            task2_train, replay_features, replay_labels, replay_original_ids,
            class_ids=ALL_IDS, draws_per_epoch=DRAW_COUNT,
            class_draw_probabilities=probabilities,
        )
        if probabilities is not None:
            planned = train_view.planned_class_draw_counts
            if not (sum(planned[class_id] for class_id in OLD_IDS) == 44_110 and
                    all(planned[class_id] == 8_822 for class_id in NEW_IDS)):
                raise ValueError("Configuration B integer exposure allocation changed")

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        model = TabularTransformerClassifier.load(best_path, device="cuda")
        if model.class_ids != OLD_IDS or model.config.learning_rate != 0.001:
            raise ValueError("Task 1 model mapping or inherited learning rate changed")
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        model.add_classes(NEW_IDS)
        initial_hash = state_hash(model)
        if expanded_initial_hash is None:
            expanded_initial_hash = initial_hash
        if initial_hash != expanded_initial_hash:
            raise ValueError("Expanded initial weights differ between configurations")
        model.config = replace(
            model.config, learning_rate=0.0001, batch_size=256, epochs=8,
            early_stopping_patience=3, training_sampler="class_balanced",
            sampler_seed=SEED, mixed_precision=False,
        )
        if not (model.class_ids == ALL_IDS and model.config.num_classes == 8 and
                (model.config.hidden_dim, model.config.num_heads, model.config.num_layers,
                 model.config.mlp_dim, model.config.batch_size, model.config.seed) ==
                (64, 4, 2, 128, 256, SEED)):
            raise ValueError("Expanded model architecture differs from the fixed protocol")

        identity = {
            "source_sha": source_sha, "task1_best_sha256": task1_best_sha,
            "task1_data_identity": task1_identity, "condition": name,
            "task2_training_rows": len(task2_train),
            "validation_rows": len(validation_all), "draws_per_epoch": DRAW_COUNT,
            "replay_features_sha256": sha256_array(replay_features),
            "replay_labels_sha256": sha256_array(replay_labels),
            "replay_original_ids_sha256": sha256_array(replay_original_ids),
            "class_draw_probabilities": (
                None if probabilities is None else {str(k): v for k, v in probabilities.items()}
            ),
            "expanded_initial_sha256": initial_hash,
            "checkpoint_selection": "validation_macro_f1",
        }
        arm_config = {
            "condition": name, "source_sha": source_sha,
            "model": asdict(model.config), "data_identity": identity,
            "stored_replay_rows": len(replay_labels),
            "stored_unique_rows_by_class": {
                str(class_id): int(np.count_nonzero(replay_labels == class_id))
                for class_id in OLD_IDS
            },
            "present_classes": list(train_view.present_class_ids),
            "planned_class_draw_counts": (
                None if train_view.planned_class_draw_counts is None else
                {str(k): v for k, v in train_view.planned_class_draw_counts.items()}
            ),
        }
        write_or_match(folder / "arm_config.json", arm_config)
        latest_path = folder / "latest.pt"
        if not latest_path.exists() and any(
            (folder / filename).exists() for filename in
            ("best.pt", "history.json", "training_manifest.json", "validation_metrics.json")
        ):
            raise RuntimeError(f"{name}: partial arm folder lacks authoritative latest.pt")
        result = train_full_disk_backed(
            model, train_view, validation_all, output_dir=folder,
            data_identity=identity, resume=latest_path.exists(),
            max_epochs_this_call=args.max_epochs_this_call,
            progress_every_batches=100, checkpoint_selection="validation_macro_f1",
            report=lambda message: print(message, flush=True),
        )
        if result["paused"]:
            raise RuntimeError(
                f"{name} paused at epoch {result['completed_epoch']}; preserve {destination} "
                "and resume it in the next Kaggle session"
            )
        history = result["history"]
        if not (history["training_rows_processed"] == DRAW_COUNT * result["completed_epoch"] and
                history["optimizer_steps"] == 276 * result["completed_epoch"] and
                history["checkpoint_selection"] == "validation_macro_f1"):
            raise ValueError(f"{name}: completed work differs from the fixed budget")
        for row in history["epochs"]:
            exposure = row["sampling_exposure"]
            if not (row["train_rows"] == exposure["draws"] == DRAW_COUNT and
                    sum(exposure["sampled_counts"].values()) == DRAW_COUNT and
                    exposure["stored_replay_rows"] == len(replay_labels) and
                    exposure["sampler_seed"] == SEED):
                raise ValueError(f"{name}: sampler exposure does not match the epoch")
            if probabilities is not None and exposure["sampled_counts"] != \
                    exposure["planned_class_draw_counts"]:
                raise ValueError("Configuration B did not realize its exact class quotas")

        latest = torch.load(latest_path, map_location="cpu", weights_only=False)
        selected_model = TabularTransformerClassifier.load(folder / "best.pt", device="cuda")
        if not (selected_model.class_ids == ALL_IDS and
                latest["best_epoch"] == result["best_epoch"] and
                latest["checkpoint_selection"] == "validation_macro_f1" and
                all(torch.equal(tensor.detach().cpu(), latest["best_state_dict"][key])
                    for key, tensor in selected_model.network.state_dict().items())):
            raise ValueError(f"{name}: best/latest checkpoint consistency failed")
        check_x, _, _ = validation_all.batch(validation_all.indices[:256])
        reloaded = TabularTransformerClassifier.load(folder / "best.pt", device="cuda")
        if not np.array_equal(selected_model.predict(check_x), reloaded.predict(check_x)):
            raise ValueError(f"{name}: bounded reload predictions differ")
        reload_record = {
            "best_epoch": result["best_epoch"], "latest_best_state_equal": True,
            "bounded_validation_predictions_identical": True,
            "best_checkpoint_sha256": sha256_file(folder / "best.pt"),
            "checkpoint_selection": "validation_macro_f1",
        }
        write_or_match(folder / "reload_verification.json", reload_record)
        arm_summaries[name] = {
            "completed_epoch": result["completed_epoch"],
            "best_epoch": result["best_epoch"],
            "best_validation_loss": result["best_validation_loss"],
            "best_validation_macro_f1": result["best_selection_value"],
            "early_stopped": result["early_stopped"],
            "training_draws": history["training_rows_processed"],
            "validation_rows_processed": history["validation_rows_processed"],
            "optimizer_steps": history["optimizer_steps"],
            "train_seconds": sum(row["train_seconds"] for row in history["epochs"]),
            "validation_seconds": sum(row["validation_seconds"] for row in history["epochs"]),
            "stored_replay_rows": len(replay_labels),
            "expanded_initial_sha256": initial_hash,
            "best_checkpoint_sha256": reload_record["best_checkpoint_sha256"],
            "exposure_by_epoch": [row["sampling_exposure"] for row in history["epochs"]],
            "sampled_counts_total": {
                str(class_id): sum(row["sampling_exposure"]["sampled_counts"][str(class_id)]
                                   for row in history["epochs"])
                for class_id in ALL_IDS
            },
        }
        write_or_match(folder / "arm_summary.json", arm_summaries[name])
        save_training_curves(history, folder)
        print(name, "complete:", result["completed_epoch"], "epochs; best",
              result["best_epoch"], "validation macro-F1", result["best_selection_value"])

    if len({arm_summaries[name]["expanded_initial_sha256"] for name in CONDITIONS}) != 1:
        raise ValueError("The two configurations did not share expanded initial weights")

    validation_records = {}
    for name in CONDITIONS:
        folder = destination / name
        metrics_path = folder / "validation_metrics.json"
        if metrics_path.exists():
            record = json.loads(metrics_path.read_text(encoding="utf-8"))
            if not (record["source_sha"] == source_sha and
                    record["best_checkpoint_sha256"] ==
                    arm_summaries[name]["best_checkpoint_sha256"]):
                raise ValueError(f"{name}: saved validation metrics are stale")
        else:
            selected_model = TabularTransformerClassifier.load(folder / "best.pt", device="cuda")
            old_metrics = evaluate_disk_backed(
                selected_model, old_validation, batch_size=256,
                class_names=class_names, benign_class_id=0,
            )
            new_metrics = evaluate_disk_backed(
                selected_model, new_validation, batch_size=256,
                class_names=class_names, benign_class_id=0,
            )
            record = {
                "condition": name, "source_sha": source_sha,
                "best_checkpoint_sha256": arm_summaries[name]["best_checkpoint_sha256"],
                "selection_partition": "fixed clean validation only",
                "metrics": combined_partition_metrics(old_metrics, new_metrics, class_names),
            }
            atomic_json(metrics_path, record)
        validation_records[name] = record

    def rank_key(name: str) -> tuple[float, float, float, str]:
        metrics = validation_records[name]["metrics"]
        return (
            -metrics["combined_validation"]["macro_f1"],
            metrics["benign_fpr"],
            -metrics["old_focused"]["macro_f1"],
            name,
        )

    chosen = sorted(CONDITIONS, key=rank_key)[0]
    selection_lock = {
        "source_sha": source_sha, "selection_rule": selection_rule,
        "selected_condition": chosen,
        "selected_checkpoint": f"{chosen}/best.pt",
        "selected_checkpoint_sha256": arm_summaries[chosen]["best_checkpoint_sha256"],
        "validation_records": validation_records,
        "test_partition_opened_before_lock": False,
    }
    write_or_match(destination / "selection_lock.json", selection_lock)
    print("Configuration locked from clean validation only:", chosen)
    for name in CONDITIONS:
        metrics = validation_records[name]["metrics"]
        print(name, "validation macro-F1", metrics["combined_validation"]["macro_f1"],
              "Benign FPR", metrics["benign_fpr"],
              "old focused F1", metrics["old_focused"]["macro_f1"],
              "new focused F1", metrics["new_focused"]["macro_f1"])

    # Test evidence is first opened only after selection_lock.json is durable.
    task1_reference = json.loads((task1_dir / "task1_test_metrics.json").read_text())
    if not (task1_reference["data_identity"] == task1_identity and
            task1_reference["best_checkpoint_sha256"] == task1_best_sha and
            task1_reference["metrics"]["class_ids"] == list(OLD_IDS)):
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
    selected_model = TabularTransformerClassifier.load(
        destination / chosen / "best.pt", device="cuda"
    )
    test_partition_metrics = {}
    for label, view in (("old", old_test), ("new", new_test)):
        target = destination / f"selected_{label}_test_metrics.json"
        if target.exists():
            record = json.loads(target.read_text(encoding="utf-8"))
            if not (record["source_sha"] == source_sha and
                    record["selected_condition"] == chosen and
                    record["selected_checkpoint_sha256"] ==
                    selection_lock["selected_checkpoint_sha256"]):
                raise ValueError(f"Saved selected {label} test metrics are stale")
        else:
            torch.cuda.synchronize()
            started = perf_counter()
            metrics = evaluate_disk_backed(
                selected_model, view, batch_size=256,
                class_names=class_names, benign_class_id=0,
            )
            torch.cuda.synchronize()
            record = {
                "source_sha": source_sha, "selected_condition": chosen,
                "selected_checkpoint_sha256": selection_lock["selected_checkpoint_sha256"],
                "partition": f"fixed {label}-class test rows",
                "inference_seconds": perf_counter() - started,
                "metrics": metrics,
            }
            atomic_json(target, record)
        test_partition_metrics[label] = record["metrics"]

    test_summary = summarize_task2_counts(
        task1_reference["metrics"],
        np.asarray(test_partition_metrics["old"]["confusion_matrix"], dtype=np.int64),
        np.asarray(test_partition_metrics["new"]["confusion_matrix"], dtype=np.int64),
        class_names=class_names,
    )
    combined_record = {
        "source_sha": source_sha, "study": "exploratory clean Task 2 retention follow-up",
        "selected_condition": chosen,
        "selected_checkpoint_sha256": selection_lock["selected_checkpoint_sha256"],
        "selection_lock_sha256": sha256_file(destination / "selection_lock.json"),
        "test_used_for_selection": False,
        "metrics": test_summary["combined_test"],
    }
    write_or_match(destination / "selected_combined_test_metrics.json", combined_record)
    final_result = {
        **combined_record,
        "task1_before": task1_reference["metrics"],
        "old_test": test_summary["old_test"],
        "new_test": test_summary["new_test"],
        "combined_test": test_summary["combined_test"],
        "old_focused_macro_f1": test_summary["old_focused_macro_f1"],
        "new_focused_macro_f1": test_summary["new_focused_macro_f1"],
        "old_focused_balanced_accuracy": test_summary["old_focused_balanced_accuracy"],
        "new_focused_balanced_accuracy": test_summary["new_focused_balanced_accuracy"],
        "benign_fpr": test_summary["benign_fpr"],
        "old_attack_to_benign": test_summary["old_attack_to_benign"],
        "forgetting": test_summary["forgetting"],
        "target_combined_accuracy": 0.85,
        "target_reached": test_summary["combined_test"]["accuracy"] >= 0.85,
        "per_class_recall_f1": {
            str(class_id): {
                "name": BROAD_CLASS_ORDER[class_id],
                "recall": test_summary["combined_test"]["per_class"][str(class_id)]["recall"],
                "f1": test_summary["combined_test"]["per_class"][str(class_id)]["f1"],
                "support": test_summary["combined_test"]["per_class"][str(class_id)]["support"],
            }
            for class_id in ALL_IDS
        },
    }
    write_or_match(destination / "selected_test_summary.json", final_result)

    plot_inputs = {
        "selected_old_test": destination / "selected_old_test_metrics.json",
        "selected_new_test": destination / "selected_new_test_metrics.json",
        "selected_combined_test": destination / "selected_combined_test_metrics.json",
    }
    for prefix, metrics_path in plot_inputs.items():
        expected = tuple(destination / f"{prefix}_{suffix}.png" for suffix in
                         ("confusion_counts", "confusion_row_normalized",
                          "per_class_recall_f1"))
        existing = [path.exists() for path in expected]
        if any(existing) and not all(existing):
            raise RuntimeError(f"Incomplete plot set for {prefix}; inspect before rerun")
        if not any(existing):
            save_result_plots(metrics_path, destination, prefix=prefix)

    study_manifest = {
        "status": "completed exploratory clean Task 2 retention follow-up",
        "source_sha": source_sha, "environment": environment,
        "prepared_input_sha256": EXPECTED_INPUT_SHA256,
        "task1_best_sha256": task1_best_sha,
        "preprocessing_sha256": task1_identity["continual_preprocessing_sha256"],
        "historical_task2_results_modified": False,
        "selection_lock_sha256": sha256_file(destination / "selection_lock.json"),
        "selected_condition": chosen,
        "selected_checkpoint_sha256": selection_lock["selected_checkpoint_sha256"],
        "combined_test_accuracy": test_summary["combined_test"]["accuracy"],
        "combined_test_macro_f1": test_summary["combined_test"]["macro_f1"],
        "target_85_percent_reached": final_result["target_reached"],
        "limitations": [
            "Historical Task 2 test results were inspected before this exploratory follow-up.",
            "Configuration B combines replay-size and sampling changes.",
            "One fixed seed cannot establish broad retention robustness.",
            "Aggregate accuracy does not hide the saved per-class recall/F1 values.",
        ],
    }
    write_or_match(destination / "study_manifest.json", study_manifest)
    print("Selected fixed-test combined accuracy:",
          test_summary["combined_test"]["accuracy"])
    print("85% target reached:", final_result["target_reached"])
    print("Per-class recall/F1:", final_result["per_class_recall_f1"])
    print("Preserve the complete output folder privately:", destination)


if __name__ == "__main__":
    main()
