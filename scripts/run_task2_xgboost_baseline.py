#!/usr/bin/env python3
"""One locked, joint-training GPU XGBoost diagnostic for fixed Task2 rows.

This is not continual learning or poisoning mitigation. No archived code is run.
The complete full-validation result fixes the model before one test pass.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import numpy as np
from numpy.lib.format import open_memmap

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_task2_clean_retention import atomic_json, sha256_array, sha256_file
from scripts.run_task2_discrimination import (
    ALL, ARRAY_SHA, INPUT_SHA, NEW, OLD, PREPROCESSING_SHA, PREP_SOURCE_SHA,
    SEED, TASK1_BEST_SHA, TASK1_SOURCE_SHA, select_replay_positions,
)
from src.data import BROAD_CLASS_ORDER, FittedPreprocessor
from src.data.large_data import large_data_paths, verify_full_partitions
from src.evaluation.streaming import ConfusionAccumulator
from src.training.disk_backed import DiskBackedFlowDataset

STUDY_ZIP_SHA = "83c3074774d3a50806d5763f3859b31bf580cad0dfb4da2270bb0f2108719381"
EVIDENCE_SHA = {
    "protocol.json": "2215ddca6aabf10321e591901b4df7ecbbf600d0861e6be82adb2f86d71ad70b",
    "replay_manifest.json": "b750c16203a54c34be5ea19493de79f9eea77e4c82950d000cfb6f4ca93c9d2e",
    "replay_original_ids_private.npy": "05c97c7add55560d561042300d3c23844704d0cace651947aa2395d7e2e5e0c7",
    "A_old70_inf20/validation_metrics.json": "8a91b9d01d39924e3da9e368144a0018b5a12a5758a6b245f1a43f19dae64651",
    "B_old75_inf15/validation_metrics.json": "1659cc6f6479069030ae092493d37aed6d19eb617121d5a77297f117a143f7da",
    "selection_lock.json": "f01362ddf8d1506d72a19c8647ea075b7e9846ad02a5700f4ab5ba9780890714",
    "findings.json": "8a4507cba8ae1ec0c7ae389c357b8a2e91aa256408dd7f6dd32b76a2780b876a",
}
MAX_ROUNDS = 500
EARLY_STOPPING = 30
BATCH = 4096
PARAMS = {
    "objective": "multi:softprob", "num_class": 8, "tree_method": "hist",
    "device": "cuda", "eval_metric": "mlogloss", "max_depth": 6,
    "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
    "max_bin": 256, "min_child_weight": 1.0, "lambda": 1.0,
    "seed": SEED, "nthread": 4,
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--task1-run-dir", type=Path, required=True)
    parser.add_argument("--discrimination-evidence", type=Path, required=True,
                        help="Original results.zip or its complete extracted folder")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _safe_member(name: str) -> bool:
    return (bool(name) and not name.startswith("/") and "\\" not in name
            and all(part not in ("", ".", "..") for part in name.split("/")))


def read_evidence(source: Path) -> tuple[dict[str, bytes], dict]:
    """Read only seven pinned data members; ZIP code/checkpoints stay inert."""
    payload: dict[str, bytes] = {}
    if source.is_file():
        if sha256_file(source) != STUDY_ZIP_SHA:
            raise ValueError("Discrimination ZIP SHA-256 differs from verified original")
        with ZipFile(source) as archive:
            for item in archive.infolist():
                mode = (item.external_attr >> 16) & 0o170000
                if (not _safe_member(item.filename.rstrip("/")) or item.flag_bits & 1
                        or mode == 0o120000):
                    raise ValueError(f"Unsafe ZIP member: {item.filename}")
            if archive.testzip() is not None:
                raise ValueError("Discrimination ZIP failed CRC verification")
            for suffix in EVIDENCE_SHA:
                matches = [name for name in archive.namelist()
                           if name.endswith("task2_discrimination_study/" + suffix)]
                if len(matches) != 1:
                    raise ValueError(f"Missing or ambiguous ZIP evidence: {suffix}")
                payload[suffix] = archive.read(matches[0])
    elif source.is_dir():
        for suffix in EVIDENCE_SHA:
            matches = list(source.rglob("task2_discrimination_study/" + suffix))
            if source.name == "task2_discrimination_study":
                matches = [source / suffix]
            matches = [path for path in matches if path.is_file() and not path.is_symlink()]
            if len(matches) != 1 or not matches[0].resolve().is_relative_to(source.resolve()):
                raise ValueError(f"Missing or ambiguous folder evidence: {suffix}")
            payload[suffix] = matches[0].read_bytes()
    else:
        raise ValueError("Attach the verified discrimination ZIP or extracted folder")
    for name, expected in EVIDENCE_SHA.items():
        if sha256(payload[name]).hexdigest() != expected:
            raise ValueError(f"Discrimination evidence changed: {name}")
    protocol = json.loads(payload["protocol.json"])
    replay = json.loads(payload["replay_manifest.json"])
    lock = json.loads(payload["selection_lock.json"])
    findings = json.loads(payload["findings.json"])
    ids = np.load(BytesIO(payload["replay_original_ids_private.npy"]), allow_pickle=False)
    if (protocol["replay"] != replay or replay["rows"] != 140_000 or
            replay["class_counts"] != {"0": 100_000, **{str(i): 10_000 for i in range(1, 5)}} or
            ids.dtype != np.int64 or ids.shape != (140_000,) or
            sha256_array(ids) != replay["ids_sha256"] or
            len(np.unique(ids)) != len(ids) or lock["selected_arm"] is not None or
            lock["eligible_count"] != 0 or "no test evaluation" not in findings["status"]):
        raise ValueError("Discrimination replay or selection provenance differs")
    comparisons = {}
    for arm in ("A_old70_inf20", "B_old75_inf15"):
        row = json.loads(payload[f"{arm}/validation_metrics.json"])
        if (row["partition"] != "full clean validation" or row["summary"]["eligible"] or
                row != lock["validation_records"][arm]):
            raise ValueError(f"Transformer validation record changed: {arm}")
        comparisons[arm] = row
    return {"replay_ids": ids, "transformer_validation": comparisons}, {
        "source": str(source), "zip_sha256": STUDY_ZIP_SHA,
        "consumed_file_sha256": EVIDENCE_SHA, "protocol": protocol,
    }


def class_weights(counts: np.ndarray) -> np.ndarray:
    """Locked train-only policy: sqrt(max_count/count), cap 5, mean-one."""
    values = np.asarray(counts, dtype=np.float64)
    if values.shape != (8,) or np.any(values <= 0):
        raise ValueError("All eight training classes must be present")
    raw = np.minimum(5.0, np.sqrt(values.max() / values))
    return raw / np.dot(raw, values / values.sum())


def check_xgboost_cuda():
    """Fail rather than silently falling back to CPU or upgrading packages."""
    import xgboost as xgb
    from packaging.version import Version

    if Version(xgb.__version__) < Version("2.0"):
        raise RuntimeError("Installed XGBoost lacks the locked device=cuda API; use a compatible Kaggle image")
    build = xgb.build_info()
    if str(build.get("USE_CUDA", "")).lower() not in ("true", "1", "on"):
        raise RuntimeError("Installed XGBoost is CPU-only; enable a GPU-compatible Kaggle image")
    probe_x = np.asarray([[0., 1.], [1., 0.], [1., 1.], [0., 0.]], dtype=np.float32)
    probe_y = np.asarray([0, 1, 1, 0], dtype=np.int32)
    probe = xgb.train({"objective": "binary:logistic", "tree_method": "hist",
                       "device": "cuda", "nthread": 1}, xgb.QuantileDMatrix(probe_x, label=probe_y),
                      num_boost_round=1, verbose_eval=False)
    runtime = json.loads(probe.save_config())["learner"]["generic_param"]["device"]
    if not runtime.startswith("cuda"):
        raise RuntimeError("XGBoost silently fell back to CPU; stop instead of upgrading blindly")
    return xgb, {"version": xgb.__version__, "build": build, "probe_device": runtime}


def materialize(view: DiskBackedFlowDataset, positions: np.ndarray, features,
                labels, offset: int = 0) -> None:
    for start in range(0, len(positions), BATCH):
        end = min(start + BATCH, len(positions))
        x, y, ids = view.batch(positions[start:end])
        if not np.array_equal(ids, positions[start:end]) or not np.isfinite(x).all():
            raise ValueError("Raw row ID or frozen transformed feature mismatch")
        features[offset + start:offset + end] = x
        labels[offset + start:offset + end] = y
        if start % (BATCH * 64) == 0 or end == len(positions):
            print(f"{view.partition}: {end:,}/{len(positions):,} transformed", flush=True)


def infer_metrics(booster, view: DiskBackedFlowDataset, xgb, *, batch_size: int = BATCH) -> dict:
    accumulator = ConfusionAccumulator(ALL)
    for start in range(0, len(view), batch_size):
        positions = view.indices[start:start + batch_size]
        x, labels, _ = view.batch(positions)
        probabilities = np.asarray(booster.predict(xgb.DMatrix(x)))
        if probabilities.shape != (len(x), 8):
            raise ValueError("XGBoost prediction/class mapping changed")
        accumulator.update(labels, probabilities.argmax(axis=1).astype(np.int64))
    return accumulator.metrics(class_names=dict(enumerate(BROAD_CLASS_ORDER)), benign_class_id=0)


def summarize(metrics: dict) -> dict:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    old_rows, new_rows = matrix[:5].sum(), matrix[5:].sum()
    return {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
            "benign_fpr": metrics["benign_false_positive"]["rate"],
            "old_accuracy": float(np.trace(matrix[:5, :5]) / old_rows),
            "new_accuracy": float(np.trace(matrix[5:, 5:]) / new_rows),
            "new_recall_each": {str(i): metrics["per_class"][str(i)]["recall"] for i in NEW}}


def main() -> None:
    args = arguments()
    for name, path in (("prepared", args.prepared_dir), ("Task1", args.task1_run_dir)):
        if not path.is_dir():
            raise ValueError(f"Attach the complete {name} directory: {path}")
    output = args.output_dir.resolve()
    if not output.is_relative_to(Path("/kaggle/working")) or output.exists():
        raise ValueError("Use a new output directory under /kaggle/working")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Enable Kaggle GPU; never launch this research run locally")
    xgb, compatibility = check_xgboost_cuda()
    evidence, evidence_meta = read_evidence(args.discrimination_evidence)
    prepared, task1 = args.prepared_dir, args.task1_run_dir
    prep = json.loads((prepared / "manifest.json").read_text())
    split = json.loads((prepared / "split_manifest.json").read_text())
    features_meta = json.loads((prepared / "features_manifest.json").read_text())
    fit = json.loads((prepared / "preprocessing_continual.json").read_text())
    foundation = json.loads((task1 / "final_manifest.json").read_text())
    config1 = json.loads((task1 / "run_config.json").read_text())
    identity = foundation["data_identity"]
    if not (prep["source_sha"] == PREP_SOURCE_SHA and prep["input_sha256"] == INPUT_SHA and
            prep["input_rows"] == split["row_count"] == features_meta["row_count"] == 9_167_581 and
            split["seed"] == SEED and split["class_order"] == list(BROAD_CLASS_ORDER) and
            split["split_codes"] == {"train": 0, "validation": 1, "test": 2} and
            features_meta["feature_names"] == fit["feature_names"] and
            len(fit["feature_names"]) == 54 and features_meta["targets_excluded"] == ["Label", "ClassLabel"] and
            fit["scope"] == "continual" and fit["fit_rows"] == 6_347_232 and
            foundation["source_sha"] == config1["source_sha"] == TASK1_SOURCE_SHA and
            foundation["data_identity"] == config1["data_identity"] and
            foundation["class_ids"] == config1["class_ids"] == list(OLD) and
            identity["preprocessing_scope"] == "Task 1 training global IDs 0-4 only" and
            identity["input_sha256"] == INPUT_SHA and
            identity["continual_preprocessing_sha256"] == PREPROCESSING_SHA):
        raise ValueError("Prepared/Task1 schema, split, preprocessing or class mapping differs")
    for name, expected in ARRAY_SHA.items():
        if sha256_file(prepared / name) != expected:
            raise ValueError(f"Prepared file SHA-256 changed: {name}")
    if (sha256_file(task1 / "preprocessing_continual.json") != PREPROCESSING_SHA or
            sha256_file(task1 / "best.pt") != TASK1_BEST_SHA or
            foundation["best_checkpoint_sha256"] != TASK1_BEST_SHA or
            verify_full_partitions(large_data_paths(prepared), 9_167_581)["counts"] != split["counts"]):
        raise ValueError("Task1 best, frozen preprocessing or fixed split differs")
    if evidence_meta["protocol"]["task1_best_sha256"] != TASK1_BEST_SHA or \
            evidence_meta["protocol"]["preprocessing_sha256"] != PREPROCESSING_SHA or \
            evidence_meta["protocol"]["prepared_input_sha256"] != INPUT_SHA:
        raise ValueError("Discrimination evidence belongs to a different foundation")
    preprocessor = FittedPreprocessor(tuple(fit["feature_names"]),
        np.asarray(fit["imputation_values"], dtype=np.float64),
        np.asarray(fit["means"], dtype=np.float64),
        np.asarray(fit["scales"], dtype=np.float64))
    old_train = DiskBackedFlowDataset(prepared, preprocessor, partition="train", class_ids=OLD)
    replay = select_replay_positions(old_train)
    if (not np.array_equal(replay, evidence["replay_ids"]) or
            sha256_array(replay) != evidence_meta["protocol"]["replay"]["positions_sha256"]):
        raise ValueError("Regenerated training-only replay differs from verified study")
    new_train = DiskBackedFlowDataset(prepared, preprocessor, partition="train", class_ids=NEW)
    validation = DiskBackedFlowDataset(prepared, preprocessor, partition="validation", class_ids=ALL)
    if (len(new_train), len(validation)) != (70_076, 1_375_139):
        raise ValueError("Fixed Task2 train or validation partition count differs")
    if np.any(old_train.splits[replay] != 0) or np.any(new_train.splits[new_train.indices] != 0):
        raise ValueError("Training rows cross fixed split boundary")
    counts = np.bincount(np.concatenate((old_train.labels[replay],
                                         new_train.labels[new_train.indices])), minlength=8)
    weights = class_weights(counts)
    protocol = {"study": "one predeclared joint-training XGBoost diagnostic baseline",
        "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "runner_file_sha256": sha256_file(Path(__file__)),
        "prepared_input_sha256": INPUT_SHA, "task1_best_sha256": TASK1_BEST_SHA,
        "preprocessing_sha256": PREPROCESSING_SHA, "class_order": list(BROAD_CLASS_ORDER),
        "feature_names": fit["feature_names"], "seed": SEED, "parameters": PARAMS,
        "max_rounds": MAX_ROUNDS, "early_stopping_rounds": EARLY_STOPPING,
        "early_stopping_metric": "validation weighted mlogloss",
        "weight_policy": "sqrt(max train count / class train count), capped at 5, normalized to train mean 1; same mapping on validation only for early stopping",
        "class_train_counts": counts.astype(int).tolist(), "class_weights": weights.tolist(),
        "old_replay_rows": len(replay), "new_training_rows": len(new_train),
        "old_replay_ids_sha256": sha256_array(replay),
        "new_training_ids_sha256": sha256_array(new_train.indices),
        "validation_rows": len(validation), "historical_test_already_inspected": True,
        "selection": "one configuration; weighted validation mlogloss chooses best boosting round; report unweighted full validation metrics",
        "practical_targets": {"benign_fpr_max": .05, "old_accuracy_min": .90,
                              "each_new_class_recall_min": .50},
        "interpretation": "joint-training diagnostic, not continual learning or poisoning defense",
        "xgboost_cuda": compatibility,
        "evidence": {"path": evidence_meta["source"],
                     "original_zip_sha256": evidence_meta["zip_sha256"],
                     "consumed_file_sha256": evidence_meta["consumed_file_sha256"]}}
    output.mkdir(parents=True)
    atomic_json(output / "protocol.json", protocol)
    train_n = len(replay) + len(new_train)
    train_x = open_memmap(output / "train_features_private.npy", mode="w+", dtype="float32", shape=(train_n, 54))
    train_y = open_memmap(output / "train_labels_private.npy", mode="w+", dtype="int32", shape=(train_n,))
    materialize(old_train, replay, train_x, train_y)
    if (sha256_array(np.asarray(train_x[:len(replay)])) != evidence_meta["protocol"]["replay"]["features_sha256"] or
            sha256_array(np.asarray(train_y[:len(replay)].astype(np.int64))) != evidence_meta["protocol"]["replay"]["labels_sha256"]):
        raise ValueError("Frozen replay features or labels differ from verified study")
    materialize(new_train, new_train.indices, train_x, train_y, len(replay))
    if not np.array_equal(np.bincount(train_y, minlength=8), counts):
        raise ValueError("Joint training labels differ from training-only manifest")
    train_x.flush(); train_y.flush()
    val_x = open_memmap(output / "validation_features_private.npy", mode="w+", dtype="float32", shape=(len(validation), 54))
    val_y = open_memmap(output / "validation_labels_private.npy", mode="w+", dtype="int32", shape=(len(validation),))
    materialize(validation, validation.indices, val_x, val_y)
    val_x.flush(); val_y.flush()
    train_matrix = xgb.QuantileDMatrix(train_x, label=train_y, weight=weights[train_y].astype(np.float32), max_bin=256)
    val_matrix = xgb.QuantileDMatrix(val_x, label=val_y, weight=weights[val_y].astype(np.float32),
                                      max_bin=256, ref=train_matrix)
    history: dict = {}
    booster = xgb.train(PARAMS, train_matrix, num_boost_round=MAX_ROUNDS,
                        evals=[(train_matrix, "train"), (val_matrix, "validation")],
                        early_stopping_rounds=EARLY_STOPPING, evals_result=history,
                        verbose_eval=25)
    best_iteration = int(booster.best_iteration)
    selected = booster[:best_iteration + 1]
    if selected.num_boosted_rounds() != best_iteration + 1:
        raise ValueError("Best boosting round was not durably selected")
    selected.save_model(str(output / "model.json"))
    atomic_json(output / "history.json", history)
    atomic_json(output / "training_manifest.json", {"best_iteration_zero_based": best_iteration,
        "completed_rounds": booster.num_boosted_rounds(),
        "selected_rounds": selected.num_boosted_rounds(),
        "model_sha256": sha256_file(output / "model.json"),
        "train_features_sha256": sha256_file(output / "train_features_private.npy"),
        "train_labels_sha256": sha256_file(output / "train_labels_private.npy"),
        "validation_features_sha256": sha256_file(output / "validation_features_private.npy"),
        "validation_labels_sha256": sha256_file(output / "validation_labels_private.npy")})
    validation_metrics = infer_metrics(selected, validation, xgb)
    transformer = {arm: {"summary": row["summary"], "metrics": row["metrics"]}
                   for arm, row in evidence["transformer_validation"].items()}
    summary = summarize(validation_metrics)
    eligible = (summary["benign_fpr"] <= .05 and summary["old_accuracy"] >= .90 and
                all(value >= .50 for value in summary["new_recall_each"].values()))
    atomic_json(output / "validation_comparison.json", {"xgboost": {"summary": summary,
        "metrics": validation_metrics, "practical_targets_met": eligible},
        "transformer_A_B": transformer,
        "comparison_caveat": "XGBoost sees all old and new labels jointly; Transformer starts from Task1. This is not a continual-learning or poisoning-defense comparison."})
    atomic_json(output / "selection_lock.json", {"model_sha256": sha256_file(output / "model.json"),
        "best_iteration_zero_based": best_iteration, "validation_summary": summary,
        "practical_targets_met": eligible, "test_opened_before_lock": False,
        "historical_test_already_inspected": True})
    print("Validation lock saved. Opening fixed test once.", flush=True)
    locked = xgb.Booster()
    locked.load_model(str(output / "model.json"))
    if (sha256_file(output / "model.json") !=
            json.loads((output / "selection_lock.json").read_text())["model_sha256"] or
            locked.num_boosted_rounds() != best_iteration + 1):
        raise ValueError("Saved validation-selected model failed reload verification")
    test = DiskBackedFlowDataset(prepared, preprocessor, partition="test", class_ids=ALL)
    if len(test) != 1_375_134:
        raise ValueError("Fixed test membership count changed")
    test_metrics = infer_metrics(locked, test, xgb)
    atomic_json(output / "test_metrics.json", {"partition": "fixed test, once after validation lock",
        "model_sha256": sha256_file(output / "model.json"), "metrics": test_metrics,
        "summary": summarize(test_metrics), "historical_test_already_inspected": True})
    atomic_json(output / "findings.json", {"validation_targets_met": eligible,
        "validation": summary, "test": summarize(test_metrics),
        "interpretation": "Joint-training diagnostic only; no continual-learning or poisoning-defense claim. Historical test has been inspected in prior studies."})
    print("Baseline complete. Review validation_comparison.json and test_metrics.json.", flush=True)


if __name__ == "__main__":
    main()
