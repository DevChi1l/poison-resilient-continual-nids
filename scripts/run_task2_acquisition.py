#!/usr/bin/env python3
"""Exploratory clean Task2 acquisition: matched sampling, with/without old-only KD.

Kaggle GPU only. Task1 best, prepared arrays, and retention evidence are immutable.
No test view is opened before the validation-only configuration lock is saved.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_task2_clean_retention import (  # noqa: E402
    atomic_json, combined_partition_metrics, sha256_array,
    sha256_file, state_hash, write_or_match,
)
from scripts.run_task2_b_poisoning import (  # noqa: E402
    EXPECTED_INPUT_SHA256, EXPECTED_PREP_SOURCE_SHA, EXPECTED_TASK1_BEST_SHA256,
    EXPECTED_TASK1_SOURCE_SHA, EXPECTED_RETENTION_ZIP_SHA256,
    RETENTION_EVIDENCE_SHA256, RETENTION_ROOT, read_verified_evidence,
    require_kaggle_directory, require_kaggle_file,
)

OLD = tuple(range(5))
NEW = (5, 6, 7)
ALL = tuple(range(8))
ARMS = ("A_revised_clean", "B_revised_old_kd")
SEED = 42
DRAWS = 141_152
LR = 0.0003
KD_WEIGHT = 0.5
KD_TEMPERATURE = 2.0


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", required=True, type=Path)
    parser.add_argument("--task1-run-dir", required=True, type=Path)
    evidence = parser.add_mutually_exclusive_group(required=True)
    evidence.add_argument("--retention-results-zip", type=Path)
    evidence.add_argument("--retention-results-dir", type=Path)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output-dir", type=Path)
    output.add_argument("--resume-run-dir", type=Path)
    parser.add_argument("--max-epochs-this-call", type=int)
    return parser.parse_args()


def output_dir(args: argparse.Namespace) -> Path:
    if args.resume_run_dir:
        previous = require_kaggle_directory(args.resume_run_dir, "resume run")
        if previous.is_relative_to(Path("/kaggle/working")):
            return previous
        target = Path("/kaggle/working") / ("task2_acquisition_resume_" +
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
        shutil.copytree(previous, target)
        return target
    target = (args.output_dir or Path("/kaggle/working") /
              ("task2_acquisition_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))).resolve()
    if not target.is_relative_to(Path("/kaggle/working")):
        raise ValueError("Output must be under /kaggle/working")
    target.mkdir(parents=True, exist_ok=True)
    return target


def class_probabilities(split_manifest: dict, order: tuple[str, ...]) -> dict[int, float]:
    """40% old by original Task1 train proportions, 40/10/10% new."""
    counts = {i: int(split_manifest["counts"][order[i]]["train"]) for i in OLD}
    total = sum(counts.values())
    values = {i: 0.4 * counts[i] / total for i in OLD}
    values.update({5: 0.4, 6: 0.1, 7: 0.1})
    if not np.isclose(sum(values.values()), 1, atol=1e-12, rtol=0):
        raise ValueError("Class probabilities do not sum to one")
    return values


def save_curves(history: dict, destination: Path) -> None:
    """Label B's train objective honestly: CE plus old-replay KD."""
    target = destination / "training_curves.png"
    if target.exists():
        return
    import matplotlib.pyplot as plt
    epochs = history["epochs"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot([e["epoch"] for e in epochs], [e["train_loss"] for e in epochs],
                 marker="o", label="train objective (CE + optional KD)")
    axes[0].plot([e["epoch"] for e in epochs], [e["validation_loss"] for e in epochs],
                 marker="o", label="clean validation CE")
    axes[0].set(xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot([e["epoch"] for e in epochs], [e["validation_macro_f1"] for e in epochs],
                 marker="o", label="clean validation macro-F1")
    axes[1].set(xlabel="Epoch", ylabel="Macro-F1", ylim=(0, 1))
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(target, dpi=160)
    plt.close(figure)


def main() -> None:
    args = arguments()
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
        raise RuntimeError("Enable Kaggle GPU; this runner never trains locally")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    prepared = require_kaggle_directory(args.prepared_dir, "prepared arrays")
    task1 = require_kaggle_directory(args.task1_run_dir, "Task1 run")
    evidence_source = (require_kaggle_file(args.retention_results_zip, "retention ZIP")
        if args.retention_results_zip else
        require_kaggle_directory(args.retention_results_dir, "retention folder"))
    destination = output_dir(args)
    for source in (prepared, task1, evidence_source):
        if destination == source or destination.is_relative_to(source):
            raise ValueError("Output must not overwrite an input")
    if shutil.disk_usage("/kaggle/working").free < 1_000_000_000:
        raise RuntimeError("At least 1 GB free /kaggle/working space required")
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         text=True).strip()
    evidence = read_verified_evidence(
        evidence_source, root=RETENTION_ROOT,
        zip_sha256=EXPECTED_RETENTION_ZIP_SHA256,
        member_sha256=RETENTION_EVIDENCE_SHA256,
    )
    historical = {key.removeprefix(RETENTION_ROOT): value for key, value in evidence.items()}
    replay_manifest = historical["large_replay_manifest.json"]
    prior = historical["selected_test_summary.json"]
    if not (replay_manifest["seed"] == SEED and
            replay_manifest["source_partition"] == "fixed training split only" and
            replay_manifest["rows_per_old_class"] == 5_000 and
            replay_manifest["total_rows"] == 25_000 and
            historical["protocol.json"]["task1_best_sha256"] == EXPECTED_TASK1_BEST_SHA256 and
            prior["combined_test"]["per_class"]["5"]["recall"] == 0.0):
        raise ValueError("Verified retention evidence has unexpected semantics")

    prep = json.loads((prepared / "manifest.json").read_text())
    split = json.loads((prepared / "split_manifest.json").read_text())
    features = json.loads((prepared / "features_manifest.json").read_text())
    continual = json.loads((prepared / "preprocessing_continual.json").read_text())
    task1_final = json.loads((task1 / "final_manifest.json").read_text())
    task1_run = json.loads((task1 / "run_config.json").read_text())
    task1_training = json.loads((task1 / "training_manifest.json").read_text())
    paths = large_data_paths(prepared)
    identity = task1_final["data_identity"]
    if not (prep["source_sha"] == EXPECTED_PREP_SOURCE_SHA and
            prep["input_sha256"] == identity["input_sha256"] == EXPECTED_INPUT_SHA256 and
            prep["input_rows"] == split["row_count"] == features["row_count"] == 9_167_581 and
            split["seed"] == SEED and split["class_order"] == list(BROAD_CLASS_ORDER) and
            split["split_codes"] == {"train": 0, "validation": 1, "test": 2} and
            len(features["feature_names"]) == 54 and
            features["targets_excluded"] == ["Label", "ClassLabel"] and
            continual["scope"] == "continual" and
            continual["feature_names"] == features["feature_names"] and
            continual["fit_rows"] == 6_347_232 and
            task1_final["source_sha"] == task1_run["source_sha"] == EXPECTED_TASK1_SOURCE_SHA and
            task1_final["data_identity"] == task1_run["data_identity"] == task1_training["data_identity"] and
            task1_final["class_ids"] == task1_run["class_ids"] == list(OLD) and
            task1_final["best_epoch"] == 9 and task1_final["completed_epoch"] == 12 and
            identity["preprocessing_scope"] == "Task 1 training global IDs 0-4 only" and
            replay_manifest["preprocessing_sha256"] == identity["continual_preprocessing_sha256"]):
        raise ValueError("Prepared/Task1 identity or class mapping changed")
    if verify_full_partitions(paths, 9_167_581)["counts"] != split["counts"]:
        raise ValueError("Fixed partition counts changed")
    for path, key in (
        (paths.raw_features, "raw_features_sha256"), (paths.labels, "labels_sha256"),
        (paths.splits, "splits_sha256"), (paths.row_ids, "row_ids_sha256"),
        (prepared / "split_manifest.json", "split_manifest_sha256"),
        (prepared / "preprocessing_continual.json", "continual_preprocessing_sha256"),
    ):
        if sha256_file(path) != identity[key]:
            raise ValueError(f"Prepared input hash changed: {key}")
    if sha256_file(task1 / "preprocessing_continual.json") != identity["continual_preprocessing_sha256"]:
        raise ValueError("Task1 frozen preprocessor copy changed")
    best = task1 / "best.pt"
    if sha256_file(best) != EXPECTED_TASK1_BEST_SHA256 == task1_final["best_checkpoint_sha256"]:
        raise ValueError("Task1 best checkpoint changed")
    checkpoint = torch.load(best, map_location="cpu", weights_only=False)
    if not (checkpoint["class_ids"] == list(OLD) and checkpoint["data_identity"] == identity and
            checkpoint["best_epoch"] == 9):
        raise ValueError("Task1 checkpoint metadata is incompatible")
    del checkpoint
    preprocessor = FittedPreprocessor(
        tuple(continual["feature_names"]),
        np.asarray(continual["imputation_values"], dtype=np.float64),
        np.asarray(continual["means"], dtype=np.float64),
        np.asarray(continual["scales"], dtype=np.float64),
    )
    task1_train = DiskBackedFlowDataset(prepared, preprocessor, partition="train", class_ids=OLD)
    positions = select_disk_replay_rows(task1_train, class_ids=OLD, per_class=5_000, seed=SEED)
    replay_x, replay_y, replay_ids = task1_train.batch(positions)
    labels_map = np.load(paths.labels, mmap_mode="r", allow_pickle=False)
    splits_map = np.load(paths.splits, mmap_mode="r", allow_pickle=False)
    raw_map = np.load(paths.raw_features, mmap_mode="r", allow_pickle=False)
    row_map = np.load(paths.row_ids, mmap_mode="r", allow_pickle=False)
    if not (replay_x.shape == (25_000, 54) and replay_x.dtype == np.float32 and
            len(np.unique(positions)) == len(np.unique(replay_ids)) == 25_000 and
            np.array_equal(replay_ids, np.asarray(replay_manifest["original_row_ids"])) and
            np.all(splits_map[positions] == 0) and
            np.array_equal(row_map[positions], replay_ids) and
            np.array_equal(labels_map[positions], replay_y) and
            np.array_equal(preprocessor.transform(raw_map[positions]), replay_x) and
            sha256_array(replay_x) == replay_manifest["features_sha256"] and
            sha256_array(replay_y) == replay_manifest["labels_sha256"] and
            sha256_array(replay_ids) == replay_manifest["original_row_ids_sha256"] and
            all(np.count_nonzero(replay_y == i) == 5_000 for i in OLD)):
        raise ValueError("25,000 training-only replay rows do not match saved manifest")

    probs = class_probabilities(split, tuple(BROAD_CLASS_ORDER))
    protocol = {
        "source_sha": source_sha, "study": "exploratory clean Task2 acquisition; historical test viewed",
        "task1_best_sha256": EXPECTED_TASK1_BEST_SHA256,
        "preprocessing_sha256": identity["continual_preprocessing_sha256"],
        "retention_evidence_sha256": EXPECTED_RETENTION_ZIP_SHA256,
        "replay_hashes": {"features": sha256_array(replay_x), "labels": sha256_array(replay_y),
                          "row_ids": sha256_array(replay_ids)},
        "shared": {"seed": SEED, "class_ids": list(ALL), "batch_size": 256,
                   "draws_per_epoch": DRAWS, "learning_rate": LR, "max_epochs": 12,
                   "patience": 4, "checkpoint_selection": "validation_macro_f1",
                   "old_sampling": "40% in original old train proportions",
                   "new_sampling": "Infiltration 40%; Webattack 10%; Portscan 10%",
                   "class_probabilities": {str(k): v for k, v in probs.items()},
                   "same_25000_replay_and_expanded_Task1_initial_weights": True,
                   "preprocessing": "frozen Task1 continual transform, once per raw row"},
        "A": {"loss": "mean eight-class cross-entropy"},
        "B": {"loss": "mean eight-class cross-entropy + 0.5 * batch-normalized T^2 KL",
              "teacher": "frozen Task1 best; old replay rows only; old five logits only",
              "distillation_weight": KD_WEIGHT, "temperature": KD_TEMPERATURE},
        "selection": "highest full clean validation macro-F1; exact tie lower Benign FPR, then higher Infiltration recall, then arm name; test once for selected only",
        "historical_B_comparison": {"combined_accuracy": prior["combined_test"]["accuracy"],
            "combined_macro_f1": prior["combined_test"]["macro_f1"],
            "old_accuracy": prior["old_test"]["accuracy"],
            "new_accuracy": prior["new_test"]["accuracy"],
            "infiltration_recall": 0.0},
        "interpretation": "Revised sampling vs historical B changes exposure, coverage, and LR together; not an isolated ablation. A vs B isolates old-only KD.",
        "quarantine_reference": {"poison_rejected": 3991, "poison_total": 4000,
            "clean_rejected": 2165, "clean_total": 21000, "poison_retained": 9,
            "scope": "training-data review, not packet/network blocking; threshold unchanged"},
    }
    write_or_match(destination / "protocol.json", protocol)
    write_or_match(destination / "environment.json", {
        "python": platform.python_version(), "numpy": np.__version__,
        "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0), "cuda_device_count": torch.cuda.device_count(),
    })
    print("Inputs, replay, fixed partitions and predeclared protocol verified", flush=True)
    task2_train = DiskBackedFlowDataset(prepared, preprocessor, partition="train", class_ids=NEW)
    validation = DiskBackedFlowDataset(prepared, preprocessor, partition="validation", class_ids=ALL)
    old_val_ids = DiskBackedFlowDataset(prepared, preprocessor, partition="validation", class_ids=OLD).indices
    new_val_ids = DiskBackedFlowDataset(prepared, preprocessor, partition="validation", class_ids=NEW).indices
    old_val = DiskBackedFlowDataset(prepared, preprocessor, partition="validation", class_ids=ALL, row_ids=old_val_ids)
    new_val = DiskBackedFlowDataset(prepared, preprocessor, partition="validation", class_ids=ALL, row_ids=new_val_ids)
    if (len(task2_train), len(validation), len(old_val), len(new_val)) != (70_076, 1_375_139, 1_360_123, 15_016):
        raise ValueError("Task2 train/validation counts changed")
    names = dict(enumerate(BROAD_CLASS_ORDER))
    first_hash = None
    summaries = {}
    records = {}
    for arm in ARMS:
        folder = destination / arm
        folder.mkdir(exist_ok=True)
        train = Task2ReplayView(task2_train, replay_x, replay_y, replay_ids,
                                class_ids=ALL, draws_per_epoch=DRAWS,
                                class_draw_probabilities=probs)
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        model = TabularTransformerClassifier.load(best, device="cuda")
        if model.class_ids != OLD:
            raise ValueError("Task1 model class mapping changed")
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        model.add_classes(NEW)
        initial_hash = state_hash(model)
        if first_hash is None:
            first_hash = initial_hash
        if initial_hash != first_hash:
            raise ValueError("Initial expanded weights differ")
        model.config = replace(model.config, learning_rate=LR, batch_size=256,
            epochs=12, early_stopping_patience=4, training_sampler="class_balanced",
            sampler_seed=SEED, mixed_precision=False)
        if model.class_ids != ALL or model.config.num_features != 54 or model.config.num_classes != 8:
            raise ValueError("Expanded model class/feature mapping changed")
        teacher = (TabularTransformerClassifier.load(best, device="cuda")
                   if arm == ARMS[1] else None)
        # Loading B's teacher constructs a module and consumes RNG. Restore the
        # same training RNG start in both arms after all construction.
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        arm_identity = {"source_sha": source_sha, "arm": arm, "task1_best_sha256": EXPECTED_TASK1_BEST_SHA256,
            "prepared_input_sha256": EXPECTED_INPUT_SHA256,
            "preprocessing_sha256": identity["continual_preprocessing_sha256"],
            "replay_features_sha256": sha256_array(replay_x),
            "replay_labels_sha256": sha256_array(replay_y),
            "replay_ids_sha256": sha256_array(replay_ids),
            "initial_expanded_sha256": initial_hash,
            "class_probabilities": {str(k): v for k, v in probs.items()},
            "distillation_weight": KD_WEIGHT if teacher is not None else 0.0}
        write_or_match(folder / "arm_config.json", {"model": asdict(model.config),
            "identity": arm_identity, "planned_class_draw_counts": train.planned_class_draw_counts})
        latest = folder / "latest.pt"
        if not latest.exists() and any((folder / name).exists() for name in
            ("best.pt", "history.json", "validation_metrics.json")):
            raise RuntimeError(f"{arm}: incomplete folder has no latest.pt")
        result = train_full_disk_backed(model, train, validation, output_dir=folder,
            data_identity=arm_identity, resume=latest.exists(),
            max_epochs_this_call=args.max_epochs_this_call, progress_every_batches=100,
            checkpoint_selection="validation_macro_f1", distillation_teacher=teacher,
            distillation_old_class_ids=OLD,
            distillation_weight=KD_WEIGHT if teacher is not None else 0.0,
            distillation_temperature=KD_TEMPERATURE)
        if result["paused"]:
            raise RuntimeError(f"{arm} paused after epoch {result['completed_epoch']}; save output and resume")
        history = result["history"]
        if not (history["training_rows_processed"] == DRAWS * result["completed_epoch"] and
                history["optimizer_steps"] == 552 * result["completed_epoch"]):
            raise ValueError("Completed training budget is inconsistent")
        for epoch in history["epochs"]:
            exposure = epoch["sampling_exposure"]
            if not (exposure["sampled_counts"] == exposure["planned_class_draw_counts"] and
                    sum(exposure["sampled_counts"].values()) == DRAWS):
                raise ValueError("Actual class exposure differs from predeclared quotas")
        reloaded = TabularTransformerClassifier.load(folder / "best.pt", device="cuda")
        latest_checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        if not (latest_checkpoint["best_epoch"] == result["best_epoch"] and
                all(torch.equal(t.detach().cpu(), latest_checkpoint["best_state_dict"][k])
                    for k, t in reloaded.network.state_dict().items())):
            raise ValueError("Best/latest checkpoint mismatch")
        summaries[arm] = {"completed_epoch": result["completed_epoch"],
            "best_epoch": result["best_epoch"], "optimizer_steps": history["optimizer_steps"],
            "initial_expanded_sha256": initial_hash,
            "best_checkpoint_sha256": sha256_file(folder / "best.pt"),
            "train_seconds": sum(row["train_seconds"] for row in history["epochs"]),
            "validation_seconds": sum(row["validation_seconds"] for row in history["epochs"]),
            "exposure_by_epoch": [row["sampling_exposure"] for row in history["epochs"]]}
        write_or_match(folder / "arm_summary.json", summaries[arm])
        save_curves(history, folder)
        metrics_path = folder / "validation_metrics.json"
        if metrics_path.exists():
            record = json.loads(metrics_path.read_text())
            if record["best_checkpoint_sha256"] != summaries[arm]["best_checkpoint_sha256"]:
                raise ValueError("Stale validation metrics")
        else:
            old_metrics = evaluate_disk_backed(reloaded, old_val, batch_size=256,
                class_names=names, benign_class_id=0)
            new_metrics = evaluate_disk_backed(reloaded, new_val, batch_size=256,
                class_names=names, benign_class_id=0)
            record = {"arm": arm, "best_checkpoint_sha256": summaries[arm]["best_checkpoint_sha256"],
                "partition": "fixed clean validation", "metrics": combined_partition_metrics(old_metrics, new_metrics, names)}
            atomic_json(metrics_path, record)
        records[arm] = record
    if len({summaries[arm]["initial_expanded_sha256"] for arm in ARMS}) != 1:
        raise ValueError("Arms did not start with identical expanded weights")
    def rank(arm: str):
        metrics = records[arm]["metrics"]
        return (-metrics["combined_validation"]["macro_f1"], metrics["benign_fpr"],
                -metrics["combined_validation"]["per_class"]["5"]["recall"], arm)
    chosen = min(ARMS, key=rank)
    lock = {"selected_arm": chosen, "selection_partition": "clean validation only",
        "selected_checkpoint_sha256": summaries[chosen]["best_checkpoint_sha256"],
        "validation_records": records, "test_opened_before_lock": False}
    write_or_match(destination / "selection_lock.json", lock)
    print("Validation-selected arm:", chosen, flush=True)
    reference = json.loads((task1 / "task1_test_metrics.json").read_text())
    if (reference["data_identity"] != identity or
            reference["best_checkpoint_sha256"] != EXPECTED_TASK1_BEST_SHA256):
        raise ValueError("Task1-before test reference incompatible")
    old_test_ids = DiskBackedFlowDataset(prepared, preprocessor, partition="test", class_ids=OLD).indices
    new_test_ids = DiskBackedFlowDataset(prepared, preprocessor, partition="test", class_ids=NEW).indices
    old_test = DiskBackedFlowDataset(prepared, preprocessor, partition="test", class_ids=ALL, row_ids=old_test_ids)
    new_test = DiskBackedFlowDataset(prepared, preprocessor, partition="test", class_ids=ALL, row_ids=new_test_ids)
    if (len(old_test), len(new_test)) != (1_360_119, 15_015):
        raise ValueError("Test split counts changed")
    selected = TabularTransformerClassifier.load(destination / chosen / "best.pt", device="cuda")
    test = {}
    for label, view in (("old", old_test), ("new", new_test)):
        target = destination / f"selected_{label}_test_metrics.json"
        if target.exists():
            saved = json.loads(target.read_text())
            if saved["selected_checkpoint_sha256"] != lock["selected_checkpoint_sha256"]:
                raise ValueError("Stale selected test metrics")
        else:
            saved = {"selected_checkpoint_sha256": lock["selected_checkpoint_sha256"],
                "partition": f"fixed {label} test", "metrics": evaluate_disk_backed(
                    selected, view, batch_size=256, class_names=names, benign_class_id=0)}
            atomic_json(target, saved)
        test[label] = saved["metrics"]
    summary = summarize_task2_counts(reference["metrics"],
        np.asarray(test["old"]["confusion_matrix"], dtype=np.int64),
        np.asarray(test["new"]["confusion_matrix"], dtype=np.int64), class_names=names)
    combined = summary["combined_test"]
    result = {"study": "exploratory clean Task2 acquisition", "selected_arm": chosen,
        "selected_checkpoint_sha256": lock["selected_checkpoint_sha256"],
        "selection_lock_sha256": sha256_file(destination / "selection_lock.json"),
        "test_used_for_selection": False, "metrics": summary,
        "per_class_recall_f1": {str(i): {key: combined["per_class"][str(i)][key]
            for key in ("recall", "f1", "support")} for i in ALL},
        "infiltration_improved_vs_historical_B": combined["per_class"]["5"]["recall"] > 0,
        "old_retention_deteriorated_vs_historical_B": summary["old_test"]["accuracy"] < prior["old_test"]["accuracy"],
        "old_focused_macro_f1": summary["old_focused_macro_f1"],
        "new_focused_macro_f1": summary["new_focused_macro_f1"],
        "benign_fpr": summary["benign_fpr"],
        "forgetting": summary["forgetting"],
        "new_accuracy_improved_vs_historical_B": summary["new_test"]["accuracy"] > prior["new_test"]["accuracy"],
        "eight_class_detection_adequate": all(combined["per_class"][str(i)]["recall"] > 0 for i in ALL),
        "historical_B": protocol["historical_B_comparison"]}
    write_or_match(destination / "selected_test_summary.json", result)
    write_or_match(destination / "selected_combined_test_metrics.json", {"metrics": combined})
    for name in ("selected_old_test", "selected_new_test", "selected_combined_test"):
        outputs = [destination / f"{name}_{suffix}.png" for suffix in
                   ("confusion_counts", "confusion_row_normalized", "per_class_recall_f1")]
        if not all(path.exists() for path in outputs):
            if any(path.exists() for path in outputs):
                raise RuntimeError("Partial plot output; preserve files and inspect before resume")
            save_result_plots(destination / f"{name}_metrics.json", destination, prefix=name)
    write_or_match(destination / "study_manifest.json", {"source_sha": source_sha,
        "selected_arm": chosen, "completed": True,
        "file_sha256": {path.name: sha256_file(path) for path in destination.glob("*.json")
                        if path.name != "study_manifest.json"}})
    print("Selected fixed-test summary:", destination / "selected_test_summary.json", flush=True)


if __name__ == "__main__":
    main()
