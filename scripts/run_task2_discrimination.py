#!/usr/bin/env python3
"""Two matched clean Task2 sampling schedules; Kaggle GPU training only.

Reuses Task1 best, fixed prepared arrays and frozen continual preprocessing.
No test access until a full-validation eligibility/selection lock is durable.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_task2_clean_retention import (  # noqa: E402
    atomic_json, save_training_curves, sha256_array, sha256_file, state_hash,
    write_or_match,
)

SEED = 42
OLD = tuple(range(5))
NEW = (5, 6, 7)
ALL = tuple(range(8))
ARMS = ("A_old70_inf20", "B_old75_inf15")
REPLAY_COUNTS = {0: 100_000, 1: 10_000, 2: 10_000, 3: 10_000, 4: 10_000}
DRAWS = 211_728
BATCH = 256
LR = 0.0003
MAX_EPOCHS = 12
PATIENCE = 4
TARGETS = {"benign_fpr_max": 0.05, "old_accuracy_min": 0.90,
           "new_recall_min_each": 0.50}

INPUT_SHA = "666af53c788c79312b421c607289cb555c2335ae51ba39b2700a357595fa8dba"
PREP_SOURCE_SHA = "5c0d519596bc6c4b99948126443635c2f7eb15f1"
TASK1_SOURCE_SHA = "e6ebfcf7a9beec4d2b8fd5245e7c62314b9fadf9"
TASK1_BEST_SHA = "aed830e152d61525c641a95ac8eded661caa8c318d9aed910abb954782454de6"
PREPROCESSING_SHA = "f7200d6aababede10ced7506bd734a850d9df2b45995e1b7839847334953695b"
ARRAY_SHA = {
    "raw_features.npy": "e797ab1b138c58d69ec436f1ef585317737f01ba6bc6c2af7a07998313de77cd",
    "labels.npy": "c0ce957d7bb2f87d338bcf97c6bf382c9d09cf7226be9ee9eb294ec8d0a72976",
    "splits.npy": "445182b087636e68089a5efd5d9645a1cc41109940d17ee778b81c935f0d2ca3",
    "row_ids.npy": "9a55f25361d1372e685653758031dd8d64df435712d1efb71f1d9e29efcde676",
    "split_manifest.json": "1f6ce4c1cb61d3b1ba77ff763439f690a040fc838118dc9ab7a74543348e208b",
    "preprocessing_continual.json": PREPROCESSING_SHA,
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", required=True, type=Path)
    parser.add_argument("--task1-run-dir", required=True, type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--output-dir", type=Path)
    group.add_argument("--resume-run-dir", type=Path)
    parser.add_argument("--max-epochs-this-call", type=int)
    return parser.parse_args()


def kaggle_dir(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir() or not any(resolved.is_relative_to(root) for root in
                                    (Path("/kaggle/input"), Path("/kaggle/working"))):
        raise ValueError(f"{label} must be a complete Kaggle input/output directory")
    return resolved


def destination(args: argparse.Namespace) -> Path:
    if args.resume_run_dir is not None:
        old = kaggle_dir(args.resume_run_dir, "resume run")
        if old.is_relative_to(Path("/kaggle/working")):
            return old
        new = Path("/kaggle/working") / ("task2_discrimination_resume_" +
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
        shutil.copytree(old, new)
        return new
    new = (args.output_dir or Path("/kaggle/working") /
        ("task2_discrimination_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))).resolve()
    if not new.is_relative_to(Path("/kaggle/working")):
        raise ValueError("Output must be under /kaggle/working")
    new.mkdir(parents=True, exist_ok=True)
    return new


def schedule(split: dict, class_order: tuple[str, ...], old_share: float) -> dict[int, float]:
    """Two locked schedules: A 70/20/5/5, B 75/15/5/5."""
    if old_share not in (0.70, 0.75):
        raise ValueError("Only the two predeclared old-class shares are allowed")
    counts = np.asarray([split["counts"][class_order[i]]["train"] for i in OLD],
                        dtype=np.float64)
    values = {i: float(old_share * counts[i] / counts.sum()) for i in OLD}
    values.update({5: 0.90 - old_share, 6: 0.05, 7: 0.05})
    if not np.isclose(sum(values.values()), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Schedule must sum to one")
    return values


def select_replay_positions(task1_train, *, seed: int = SEED) -> np.ndarray:
    """Select distinct original training IDs per old class, no feature reads."""
    if task1_train.partition != "train" or task1_train.class_ids != OLD:
        raise ValueError("Replay source must be original old-class training view")
    rng = np.random.default_rng(seed)
    positions = np.asarray(task1_train.indices)
    labels = task1_train.labels
    chosen = []
    for class_id, count in REPLAY_COUNTS.items():
        bucket = positions[labels[positions] == class_id]
        if len(bucket) < count:
            raise ValueError(f"Not enough original training rows for class {class_id}")
        chosen.append(np.sort(rng.choice(bucket, size=count, replace=False)))
    selected = np.concatenate(chosen)
    if (len(selected) != sum(REPLAY_COUNTS.values()) or
            len(np.unique(selected)) != len(selected) or
            np.any(task1_train.splits[selected] != 0) or
            any(np.count_nonzero(labels[selected] == i) != REPLAY_COUNTS[i] for i in OLD)):
        raise ValueError("Replay selection is not distinct and training-only")
    return selected


def replay_batches(task1_train, positions: np.ndarray, *, width: int = 54):
    """Bound feature materialization to 4,096 rows per disk-backed batch."""
    features = np.empty((len(positions), width), dtype=np.float32)
    labels = np.empty(len(positions), dtype=np.int64)
    ids = np.empty(len(positions), dtype=np.int64)
    for start in range(0, len(positions), 4_096):
        end = min(start + 4_096, len(positions))
        X, y, original = task1_train.batch(positions[start:end])
        features[start:end], labels[start:end], ids[start:end] = X, y, original
    if not np.isfinite(features).all() or not np.array_equal(ids, positions):
        raise ValueError("Frozen replay features/IDs are invalid")
    return features, labels, ids


def summarize(metrics: dict) -> dict:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    old_rows, new_rows = int(matrix[:5].sum()), int(matrix[5:].sum())
    if matrix.shape != (8, 8) or min(old_rows, new_rows) == 0:
        raise ValueError("Expected complete eight-class validation counts")
    recalls = {str(i): metrics["per_class"][str(i)]["recall"] for i in NEW}
    fpr = metrics["benign_false_positive"]["rate"]
    old_accuracy = float(np.trace(matrix[:5, :5]) / old_rows)
    new_accuracy = float(np.trace(matrix[5:, 5:]) / new_rows)
    return {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
        "benign_fpr": fpr, "old_accuracy": old_accuracy,
        "new_accuracy": new_accuracy, "new_recall": recalls,
        "eligible": (fpr <= TARGETS["benign_fpr_max"] and
            old_accuracy >= TARGETS["old_accuracy_min"] and
            all(value >= TARGETS["new_recall_min_each"] for value in recalls.values()))}


def select(rows: dict[str, dict]) -> str | None:
    eligible = [name for name in ARMS if rows[name]["summary"]["eligible"]]
    if not eligible:
        return None
    return min(eligible, key=lambda name: (
        -rows[name]["summary"]["macro_f1"],
        rows[name]["summary"]["benign_fpr"],
        -rows[name]["summary"]["new_recall"]["5"], name))


def main() -> None:
    args = arguments()
    if args.max_epochs_this_call is not None and args.max_epochs_this_call < 1:
        raise ValueError("max-epochs-this-call must be positive")
    import torch
    from src.data import BROAD_CLASS_ORDER, FittedPreprocessor
    from src.data.large_data import large_data_paths, verify_full_partitions
    from src.evaluation.plots import save_result_plots
    from src.evaluation.streaming import evaluate_disk_backed, summarize_task2_counts
    from src.models import TabularTransformerClassifier
    from src.training.disk_backed import DiskBackedFlowDataset
    from src.training.full_run import train_full_disk_backed
    from src.training.task2_replay import Task2ReplayView

    if not torch.cuda.is_available():
        raise RuntimeError("Enable Kaggle GPU; do not train this study locally")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    prepared = kaggle_dir(args.prepared_dir, "prepared input")
    task1 = kaggle_dir(args.task1_run_dir, "Task1 foundation")
    output = destination(args)
    if any(output == source or output.is_relative_to(source) for source in (prepared, task1)):
        raise ValueError("Output must not overwrite read-only inputs")
    if shutil.disk_usage("/kaggle/working").free < 1_000_000_000:
        raise RuntimeError("At least 1 GB free /kaggle/working is required")
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         text=True).strip()
    prep = json.loads((prepared / "manifest.json").read_text())
    split = json.loads((prepared / "split_manifest.json").read_text())
    feature_meta = json.loads((prepared / "features_manifest.json").read_text())
    continual = json.loads((prepared / "preprocessing_continual.json").read_text())
    foundation = json.loads((task1 / "final_manifest.json").read_text())
    config1 = json.loads((task1 / "run_config.json").read_text())
    identity = foundation["data_identity"]
    if not (prep["source_sha"] == PREP_SOURCE_SHA and prep["input_sha256"] == INPUT_SHA and
            prep["completed_preparation_and_benchmark_only"] is True and
            prep["input_rows"] == split["row_count"] == feature_meta["row_count"] == 9_167_581 and
            split["seed"] == SEED and split["class_order"] == list(BROAD_CLASS_ORDER) and
            split["split_codes"] == {"train": 0, "validation": 1, "test": 2} and
            len(feature_meta["feature_names"]) == 54 and
            feature_meta["feature_names"] == continual["feature_names"] and
            feature_meta["targets_excluded"] == ["Label", "ClassLabel"] and
            continual["scope"] == "continual" and continual["fit_rows"] == 6_347_232 and
            foundation["source_sha"] == config1["source_sha"] == TASK1_SOURCE_SHA and
            foundation["data_identity"] == config1["data_identity"] and
            foundation["class_ids"] == config1["class_ids"] == list(OLD) and
            identity["preprocessing_scope"] == "Task 1 training global IDs 0-4 only" and
            identity["input_sha256"] == INPUT_SHA and
            identity["continual_preprocessing_sha256"] == PREPROCESSING_SHA):
        raise ValueError("Prepared/Task1 data identity, class order, or preprocessing changed")
    for name, expected in ARRAY_SHA.items():
        if sha256_file(prepared / name) != expected or identity.get({
            "raw_features.npy": "raw_features_sha256", "labels.npy": "labels_sha256",
            "splits.npy": "splits_sha256", "row_ids.npy": "row_ids_sha256",
            "split_manifest.json": "split_manifest_sha256",
            "preprocessing_continual.json": "continual_preprocessing_sha256"}[name]) != expected:
            raise ValueError(f"Prepared array/Task1 hash mismatch: {name}")
    if sha256_file(task1 / "preprocessing_continual.json") != PREPROCESSING_SHA:
        raise ValueError("Task1 frozen preprocessor copy changed")
    if verify_full_partitions(large_data_paths(prepared), 9_167_581)["counts"] != split["counts"]:
        raise ValueError("Fixed partition membership changed")
    best1 = task1 / "best.pt"
    if sha256_file(best1) != TASK1_BEST_SHA == foundation["best_checkpoint_sha256"]:
        raise ValueError("Task1 best checkpoint hash changed")
    checked = torch.load(best1, map_location="cpu", weights_only=False)
    if checked["class_ids"] != list(OLD) or checked["data_identity"] != identity:
        raise ValueError("Task1 checkpoint class/data mapping changed")
    del checked
    preprocessor = FittedPreprocessor(tuple(continual["feature_names"]),
        np.asarray(continual["imputation_values"], dtype=np.float64),
        np.asarray(continual["means"], dtype=np.float64),
        np.asarray(continual["scales"], dtype=np.float64))
    old_train = DiskBackedFlowDataset(prepared, preprocessor, partition="train", class_ids=OLD)
    positions = select_replay_positions(old_train)
    replay_x, replay_y, replay_ids = replay_batches(old_train, positions)
    if not (np.all(old_train.splits[positions] == 0) and
            np.array_equal(old_train.labels[positions], replay_y) and
            all(np.count_nonzero(replay_y == i) == count for i, count in REPLAY_COUNTS.items())):
        raise ValueError("Replay labels or split isolation changed")
    replay_ids_path = output / "replay_original_ids_private.npy"
    if replay_ids_path.exists():
        saved_ids = np.load(replay_ids_path, mmap_mode="r", allow_pickle=False)
        if not np.array_equal(saved_ids, replay_ids):
            raise ValueError("Saved private replay IDs differ on resume")
    else:
        np.save(replay_ids_path, replay_ids)
    replay_identity = {"positions_sha256": sha256_array(positions),
        "features_sha256": sha256_array(replay_x), "labels_sha256": sha256_array(replay_y),
        "ids_sha256": sha256_array(replay_ids), "rows": len(replay_ids),
        "unique_original_rows": len(np.unique(replay_ids)),
        "class_counts": {str(i): int(np.count_nonzero(replay_y == i)) for i in OLD},
        "source_partition": "fixed original training only",
        "single_frozen_preprocessing_pass": True, "seed": SEED}
    write_or_match(output / "replay_manifest.json", replay_identity)
    probabilities = {
        ARMS[0]: schedule(split, tuple(BROAD_CLASS_ORDER), 0.70),
        ARMS[1]: schedule(split, tuple(BROAD_CLASS_ORDER), 0.75),
    }
    protocol = {"source_sha": source_sha, "study": "exploratory clean Task2 discrimination",
        "historical_test_already_inspected": True,
        "task1_best_sha256": TASK1_BEST_SHA, "prepared_input_sha256": INPUT_SHA,
        "preprocessing_sha256": PREPROCESSING_SHA,
        "replay": replay_identity, "arms": list(ARMS),
        "common": {"architecture": {k: config1["model"][k] for k in
            ("num_features", "hidden_dim", "num_heads", "num_layers", "mlp_dim", "dropout", "weight_decay")},
            "seed": SEED, "batch_size": BATCH, "learning_rate": LR,
            "draws_per_epoch": DRAWS, "optimizer_steps_per_epoch": 828,
            "max_epochs": MAX_EPOCHS, "early_stopping_patience": PATIENCE,
            "checkpoint_selection": "validation_macro_f1",
            "replay_rows": sum(REPLAY_COUNTS.values()),
            "replay_per_class": {str(k): v for k, v in REPLAY_COUNTS.items()},
            "preprocessing": "frozen Task1 continual transform; replay transformed once"},
        "schedules": {arm: {str(k): v for k, v in values.items()}
                      for arm, values in probabilities.items()},
        "acceptance_targets": TARGETS,
        "selection": "among eligible, highest full clean validation macro-F1; ties lower Benign FPR, higher Infiltration recall, arm name; no eligible means no test",
        "test_policy": "only validation-selected checkpoint, once on fixed test after durable lock",
        "hypotheses": ["100k distinct Benign replay may improve boundary coverage",
            "moderating Infiltration exposure may reduce Benign-to-Infiltration false positives",
            "sampled cross-label feature collisions may limit achievable discrimination"],
        "hypotheses_are_not_confirmed_causes": True}
    write_or_match(output / "protocol.json", protocol)
    print("Prepared/Task1 hashes, fixed splits and expanded replay verified", flush=True)
    new_train = DiskBackedFlowDataset(prepared, preprocessor, partition="train", class_ids=NEW)
    validation = DiskBackedFlowDataset(prepared, preprocessor, partition="validation", class_ids=ALL)
    if (len(new_train), len(validation)) != (70_076, 1_375_139):
        raise ValueError("Task2 train/validation split counts changed")
    class_names = dict(enumerate(BROAD_CLASS_ORDER))
    initial_hash = None
    records = {}
    for arm in ARMS:
        folder = output / arm
        folder.mkdir(exist_ok=True)
        train = Task2ReplayView(new_train, replay_x, replay_y, replay_ids,
            class_ids=ALL, draws_per_epoch=DRAWS,
            class_draw_probabilities=probabilities[arm])
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        model = TabularTransformerClassifier.load(best1, device="cuda")
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        model.add_classes(NEW)
        candidate_hash = state_hash(model)
        if initial_hash is None:
            initial_hash = candidate_hash
        if candidate_hash != initial_hash:
            raise ValueError("Two arms must share identical expanded Task1 weights")
        model.config = replace(model.config, learning_rate=LR, batch_size=BATCH,
            epochs=MAX_EPOCHS, early_stopping_patience=PATIENCE,
            training_sampler="class_balanced", sampler_seed=SEED,
            mixed_precision=False)
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        data_identity = {"source_sha": source_sha, "arm": arm,
            "task1_best_sha256": TASK1_BEST_SHA, "preprocessing_sha256": PREPROCESSING_SHA,
            "prepared_input_sha256": INPUT_SHA, "replay": replay_identity,
            "probabilities": {str(k): v for k, v in probabilities[arm].items()},
            "expanded_initial_sha256": initial_hash}
        write_or_match(folder / "arm_config.json", {"model": asdict(model.config),
            "data_identity": data_identity,
            "planned_class_draw_counts": {str(k): v for k, v in train.planned_class_draw_counts.items()}})
        latest = folder / "latest.pt"
        if not latest.exists() and any((folder / name).exists() for name in
                                  ("best.pt", "history.json", "validation_metrics.json")):
            raise RuntimeError(f"{arm}: partial run has no authoritative latest.pt")
        result = train_full_disk_backed(model, train, validation, output_dir=folder,
            data_identity=data_identity, resume=latest.exists(),
            max_epochs_this_call=args.max_epochs_this_call, progress_every_batches=200,
            checkpoint_selection="validation_macro_f1")
        if result["paused"]:
            raise RuntimeError(f"{arm} paused after {result['completed_epoch']} epochs; save complete output and resume")
        history = result["history"]
        if (history["optimizer_steps"] != 828 * result["completed_epoch"] or
                history["training_rows_processed"] != DRAWS * result["completed_epoch"]):
            raise ValueError(f"{arm}: actual completed budget differs")
        for row in history["epochs"]:
            exposure = row["sampling_exposure"]
            if not (exposure["sampled_counts"] == exposure["planned_class_draw_counts"] ==
                    {str(k): v for k, v in train.planned_class_draw_counts.items()} and
                    exposure["draws"] == DRAWS and exposure["stored_replay_rows"] == 140_000):
                raise ValueError(f"{arm}: sampler exposure differs from locked schedule")
        selected = TabularTransformerClassifier.load(folder / "best.pt", device="cuda")
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if not (saved["best_epoch"] == result["best_epoch"] and
                all(torch.equal(value.detach().cpu(), saved["best_state_dict"][key])
                    for key, value in selected.network.state_dict().items())):
            raise ValueError(f"{arm}: durable best/latest checkpoint mismatch")
        best_hash = sha256_file(folder / "best.pt")
        summary = {"completed_epoch": result["completed_epoch"],
            "best_epoch": result["best_epoch"], "optimizer_steps": history["optimizer_steps"],
            "initial_expanded_sha256": initial_hash, "best_checkpoint_sha256": best_hash,
            "exposure_by_epoch": [row["sampling_exposure"] for row in history["epochs"]],
            "train_seconds": sum(row["train_seconds"] for row in history["epochs"]),
            "validation_seconds": sum(row["validation_seconds"] for row in history["epochs"])}
        write_or_match(folder / "arm_summary.json", summary)
        save_training_curves(history, folder)
        metrics_path = folder / "validation_metrics.json"
        if metrics_path.exists():
            record = json.loads(metrics_path.read_text())
            if record["best_checkpoint_sha256"] != best_hash:
                raise ValueError(f"{arm}: saved validation record is stale")
        else:
            metrics = evaluate_disk_backed(selected, validation, batch_size=BATCH,
                class_names=class_names, benign_class_id=0)
            record = {"arm": arm, "best_checkpoint_sha256": best_hash,
                "partition": "full clean validation", "metrics": metrics,
                "summary": summarize(metrics)}
            atomic_json(metrics_path, record)
        records[arm] = record
        print(arm, "validation macro-F1", record["summary"]["macro_f1"],
              "Benign FPR", record["summary"]["benign_fpr"], flush=True)
    chosen = select(records)
    lock = {"selected_arm": chosen, "eligible_count": sum(records[a]["summary"]["eligible"] for a in ARMS),
        "validation_records": records, "acceptance_targets": TARGETS,
        "test_opened_before_lock": False}
    write_or_match(output / "selection_lock.json", lock)
    if chosen is None:
        best_tradeoff = min(ARMS, key=lambda arm: -records[arm]["summary"]["macro_f1"])
        write_or_match(output / "best_validation_tradeoff_metrics.json",
                       {"metrics": records[best_tradeoff]["metrics"]})
        save_result_plots(output / "best_validation_tradeoff_metrics.json", output,
                          prefix="best_ineligible_validation")
        write_or_match(output / "findings.json", {"status": "NO ELIGIBLE MODEL; no test evaluation",
            "best_validation_macro_f1_arm": best_tradeoff,
            "validation_tradeoffs": {arm: records[arm]["summary"] for arm in ARMS},
            "historical_test_already_inspected": True,
            "interpretation": "Separate new-class recall targets or Benign/old targets failed; no eight-class success claim."})
        print("No validation-eligible model; fixed test remains unopened", flush=True)
        return
    # Selection is durable before constructing any test view or reading Task1 test reference.
    reference = json.loads((task1 / "task1_test_metrics.json").read_text())
    if reference["data_identity"] != identity or reference["best_checkpoint_sha256"] != TASK1_BEST_SHA:
        raise ValueError("Task1-before test reference changed")
    test = DiskBackedFlowDataset(prepared, preprocessor, partition="test", class_ids=ALL)
    if len(test) != 1_375_134:
        raise ValueError("Fixed test split count changed")
    selected = TabularTransformerClassifier.load(output / chosen / "best.pt", device="cuda")
    test_metrics = evaluate_disk_backed(selected, test, batch_size=BATCH,
        class_names=class_names, benign_class_id=0)
    counts = np.asarray(test_metrics["confusion_matrix"], dtype=np.int64)
    old = counts.copy(); old[5:] = 0
    new = counts.copy(); new[:5] = 0
    comparison = summarize_task2_counts(reference["metrics"], old, new,
                                        class_names=class_names)
    result = {"selected_arm": chosen,
        "selected_checkpoint_sha256": records[chosen]["best_checkpoint_sha256"],
        "selection_lock_sha256": sha256_file(output / "selection_lock.json"),
        "partition": "fixed test, one selected checkpoint pass",
        "summary": summarize(test_metrics), "metrics": test_metrics,
        "forgetting": comparison["forgetting"],
        "old_attack_to_benign": comparison["old_attack_to_benign"]}
    write_or_match(output / "selected_test_metrics.json", result)
    save_result_plots(output / "selected_test_metrics.json", output, prefix="selected_test")
    write_or_match(output / "findings.json", {"status": "validation-eligible model; exploratory selected test",
        "selected_arm": chosen, "selected_validation": records[chosen]["summary"],
        "selected_test": result["summary"],
        "per_class_test": test_metrics["per_class"],
        "historical_test_already_inspected": True,
        "interpretation": "One seed and one selected test pass; inspect Infiltration precision/recall and Benign FPR before any success claim."})
    print("Validation-selected model scored once on fixed test:", chosen, flush=True)


if __name__ == "__main__":
    main()
