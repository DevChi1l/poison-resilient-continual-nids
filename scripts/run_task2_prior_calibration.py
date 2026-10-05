#!/usr/bin/env python3
"""Validation-only prior correction of the completed A/B acquisition checkpoints.

Kaggle GPU inference only. Never trains, edits source evidence, or opens the
fixed test partition unless a predeclared validation-eligible choice is locked.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_task2_clean_retention import atomic_json, sha256_file, write_or_match  # noqa: E402
from src.evaluation.prior_correction import (  # noqa: E402
    STRENGTHS, TARGETS, corrected_predictions, prior_log_shift,
    select_candidate, summarize_metrics,
)

ZIP_SHA256 = "6fc447d8fd394a411e25f482bf454cc1ae62d7eec7fbc1e0bcdc2fec08a21cf7"
ZIP_ROOT = "task2_clean_acquisition_study/"
ACQUISITION_SOURCE_SHA = "095c7b639d54eaf196a3bb324ce147a7c43c12e0"
INPUT_SHA256 = "666af53c788c79312b421c607289cb555c2335ae51ba39b2700a357595fa8dba"
PREP_SOURCE_SHA = "5c0d519596bc6c4b99948126443635c2f7eb15f1"
TASK1_BEST_SHA256 = "aed830e152d61525c641a95ac8eded661caa8c318d9aed910abb954782454de6"
PREPROCESSING_SHA256 = "f7200d6aababede10ced7506bd734a850d9df2b45995e1b7839847334953695b"
PREPARED_ARRAY_SHA256 = {
    "raw_features.npy": "e797ab1b138c58d69ec436f1ef585317737f01ba6bc6c2af7a07998313de77cd",
    "labels.npy": "c0ce957d7bb2f87d338bcf97c6bf382c9d09cf7226be9ee9eb294ec8d0a72976",
    "splits.npy": "445182b087636e68089a5efd5d9645a1cc41109940d17ee778b81c935f0d2ca3",
    "row_ids.npy": "9a55f25361d1372e685653758031dd8d64df435712d1efb71f1d9e29efcde676",
    "split_manifest.json": "1f6ce4c1cb61d3b1ba77ff763439f690a040fc838118dc9ab7a74543348e208b",
    "preprocessing_continual.json": PREPROCESSING_SHA256,
}
ARMS = ("A_revised_clean", "B_revised_old_kd")
CLASS_IDS = tuple(range(8))
EVIDENCE_SHA256 = {
    "protocol.json": "734c76c7ff5182c1efb0d7cca1cdc92fb68df3a76a9efab1e0e88def63405f0a",
    "selection_lock.json": "df3a168c051b3832abab43a0ce8134d76a1eb8fb90ceabeadf28e2551bcfa813",
    "study_manifest.json": "e05593a524489c804bef8527fe6674e0cefafd5e86bd15e88b5510772b2e20ec",
    "A_revised_clean/arm_config.json": "2b5b6503476d92e9bf4295857cbfd263a05f276c01b3bc1f7c227bec569509b2",
    "A_revised_clean/arm_summary.json": "401e0d5f2121de7a6a464bf74ee9f1b09f87d818858509e46a6b78654e6fdae8",
    "A_revised_clean/validation_metrics.json": "db2e866166b485c5dd56d701d0ca92a174f4380694fd6e5cfe1cb6bc82f9453f",
    "A_revised_clean/training_manifest.json": "1ceeff6a7fecdad6605b8a7927f2a424029d0da60b49b9adaf78a049b9e63989",
    "A_revised_clean/best.pt": "6b7eff361a3eeebf8d13963c8be4cff6c37a9c35207ae9e392eef9e914d08a67",
    "B_revised_old_kd/arm_config.json": "900e811ad0d8d7759c7663a78630d74ebaf44b15c4e336473b9967e7051bc9fd",
    "B_revised_old_kd/arm_summary.json": "4f80f6b75a3858c19d625997a7c7667254e677454e020c50d3576e051ddc9d1e",
    "B_revised_old_kd/validation_metrics.json": "945c0dc124ce9fb01e6980e066fea57c904084b77285de3d97f58e0185098e0b",
    "B_revised_old_kd/training_manifest.json": "81a1bf6c9d7cd3f370a7b9d4113090f0cf8c6eadb10e01503b94e8e639512bef",
    "B_revised_old_kd/best.pt": "72fc99d0efd8970aba9dc87e68ca63e41fa49080e081dafee7b8a2098a94016b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--acquisition-results-zip", type=Path)
    source.add_argument("--acquisition-results-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _safe_zip(archive: ZipFile) -> None:
    for info in archive.infolist():
        name = PurePosixPath(info.filename)
        if (name.is_absolute() or ".." in name.parts or "\\" in info.filename or
                stat.S_ISLNK(info.external_attr >> 16) or info.flag_bits & 1):
            raise ValueError(f"Unsafe or encrypted archive member: {info.filename}")


def read_verified_acquisition(source: Path) -> dict[str, bytes]:
    """Verify intact ZIP or each consumed extracted file, including checkpoints."""
    source = Path(source)
    if source.is_file():
        if sha256_file(source) != ZIP_SHA256:
            raise ValueError("Acquisition ZIP SHA-256 differs from verified original")
        with ZipFile(source) as archive:
            _safe_zip(archive)
            if archive.testzip() is not None:
                raise ValueError("Acquisition ZIP CRC check failed")
            available = set(archive.namelist())
            if not {ZIP_ROOT + name for name in EVIDENCE_SHA256} <= available:
                raise ValueError("Acquisition ZIP lacks required saved evidence")
            values = {name: archive.read(ZIP_ROOT + name) for name in EVIDENCE_SHA256}
    elif source.is_dir():
        values = {}
        root = source.resolve()
        for name in EVIDENCE_SHA256:
            path = source / name
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
                raise ValueError(f"Unsafe or missing acquisition evidence: {name}")
            values[name] = path.read_bytes()
    else:
        raise FileNotFoundError(f"Acquisition evidence not found: {source}")
    for name, expected in EVIDENCE_SHA256.items():
        if hashlib.sha256(values[name]).hexdigest() != expected:
            raise ValueError(f"Acquisition evidence SHA-256 mismatch: {name}")
    return values


def verify_acquisition(values: dict[str, bytes], split_manifest: dict) -> dict:
    """Check saved class mapping, frozen state, quotas and validation record."""
    load = lambda name: json.loads(values[name])
    protocol = load("protocol.json")
    lock = load("selection_lock.json")
    study = load("study_manifest.json")
    if not (protocol["source_sha"] == study["source_sha"] == ACQUISITION_SOURCE_SHA and
            protocol["task1_best_sha256"] == TASK1_BEST_SHA256 and
            protocol["preprocessing_sha256"] == PREPROCESSING_SHA256 and
            protocol["shared"]["class_ids"] == list(CLASS_IDS) and
            protocol["shared"]["draws_per_epoch"] == 141_152 and
            protocol["shared"]["checkpoint_selection"] == "validation_macro_f1" and
            lock["selection_partition"] == "clean validation only" and
            lock["selected_arm"] == "A_revised_clean" and
            lock["selected_checkpoint_sha256"] == EVIDENCE_SHA256["A_revised_clean/best.pt"] and
            study["completed"] is True):
        raise ValueError("Acquisition protocol or provenance differs from verified run")
    probabilities = np.asarray([protocol["shared"]["class_probabilities"][str(i)]
                                for i in CLASS_IDS], dtype=np.float64)
    for arm in ARMS:
        config = load(f"{arm}/arm_config.json")
        summary = load(f"{arm}/arm_summary.json")
        training = load(f"{arm}/training_manifest.json")
        recorded_validation = load(f"{arm}/validation_metrics.json")
        identity = config["identity"]
        planned = config["planned_class_draw_counts"]
        if not (config["model"]["num_features"] == 54 and
                config["model"]["num_classes"] == 8 and
                config["model"]["training_sampler"] == "class_balanced" and
                identity["source_sha"] == ACQUISITION_SOURCE_SHA and
                identity["arm"] == arm and
                identity["task1_best_sha256"] == TASK1_BEST_SHA256 and
                identity["prepared_input_sha256"] == INPUT_SHA256 and
                identity["preprocessing_sha256"] == PREPROCESSING_SHA256 and
                identity["class_probabilities"] == protocol["shared"]["class_probabilities"] and
                training["class_ids"] == list(CLASS_IDS) and
                training["config"] == config["model"] and
                training["data_identity"] == identity and
                training["checkpoint_selection"] == "validation_macro_f1" and
                summary["best_checkpoint_sha256"] == EVIDENCE_SHA256[f"{arm}/best.pt"] and
                summary["completed_epoch"] == training["completed_epoch"] == len(summary["exposure_by_epoch"]) and
                summary["best_epoch"] == training["best_epoch"] and
                recorded_validation["best_checkpoint_sha256"] == summary["best_checkpoint_sha256"] and
                len(planned) == 8 and sum(planned.values()) == 141_152):
            raise ValueError(f"{arm}: checkpoint, mapping, or training provenance mismatch")
        for exposure in summary["exposure_by_epoch"]:
            if not (exposure["draws"] == 141_152 and
                    exposure["sampler_seed"] == 42 and
                    exposure["sampled_counts"] == planned == exposure["planned_class_draw_counts"] and
                    exposure["expected_class_probabilities"] == identity["class_probabilities"] and
                    exposure["stored_replay_rows"] == 25_000):
                raise ValueError(f"{arm}: actual sampling differed from recorded quotas")
            for class_id in CLASS_IDS:
                maximum = (5_000 if class_id < 5 else
                           split_manifest["counts"][split_manifest["class_order"][class_id]]["train"])
                if not 0 < exposure["unique_rows_drawn_by_class"][str(class_id)] <= maximum:
                    raise ValueError(f"{arm}: unique-row exposure exceeds training-only pool")
    return {"protocol": protocol, "probabilities": probabilities}


def require_kaggle_path(path: Path, *, file: bool) -> Path:
    resolved = path.resolve()
    if (not resolved.is_relative_to(Path("/kaggle/input")) and
            not resolved.is_relative_to(Path("/kaggle/working"))):
        raise ValueError("Inputs must be attached under /kaggle/input or /kaggle/working")
    if not (resolved.is_file() if file else resolved.is_dir()):
        raise FileNotFoundError(path)
    return resolved


def evaluate_grid(model, view, offsets: np.ndarray, strengths: tuple[float, ...],
                  class_names: dict[int, str]) -> dict[float, dict]:
    import torch
    from src.evaluation.streaming import ConfusionAccumulator

    model.network.eval()
    accumulators = {strength: ConfusionAccumulator(CLASS_IDS) for strength in strengths}
    with torch.inference_mode():
        for start in range(0, len(view), 256):
            features, labels, _ = view.batch(view.indices[start:start + 256])
            inputs = torch.as_tensor(features, dtype=torch.float32, device=model.device)
            logits = model.network(inputs).float().cpu().numpy()
            for strength in strengths:
                predictions = corrected_predictions(logits, offsets, strength)
                accumulators[strength].update(labels, predictions)
    return {strength: accumulator.metrics(class_names=class_names, benign_class_id=0)
            for strength, accumulator in accumulators.items()}


def main() -> None:
    args = parse_args()
    import torch
    from src.data import BROAD_CLASS_ORDER, FittedPreprocessor
    from src.data.large_data import large_data_paths, verify_full_partitions
    from src.evaluation.plots import save_result_plots
    from src.models import TabularTransformerClassifier
    from src.training.disk_backed import DiskBackedFlowDataset

    if not torch.cuda.is_available():
        raise RuntimeError("Enable Kaggle GPU for bounded full-validation inference")
    prepared = require_kaggle_path(args.prepared_dir, file=False)
    acquisition = require_kaggle_path(
        args.acquisition_results_zip or args.acquisition_results_dir,
        file=args.acquisition_results_zip is not None,
    )
    output = (args.output_dir or Path("/kaggle/working") /
              ("task2_prior_calibration_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))).resolve()
    if not output.is_relative_to(Path("/kaggle/working")) or output == prepared or output == acquisition:
        raise ValueError("Output must be separate under /kaggle/working")
    output.mkdir(parents=True, exist_ok=True)
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         text=True).strip()
    prep = json.loads((prepared / "manifest.json").read_text())
    split = json.loads((prepared / "split_manifest.json").read_text())
    features = json.loads((prepared / "features_manifest.json").read_text())
    continual = json.loads((prepared / "preprocessing_continual.json").read_text())
    if not (prep["source_sha"] == PREP_SOURCE_SHA and prep["input_sha256"] == INPUT_SHA256 and
            prep["completed_preparation_and_benchmark_only"] is True and
            prep["input_rows"] == split["row_count"] == features["row_count"] == 9_167_581 and
            split["seed"] == 42 and split["class_order"] == list(BROAD_CLASS_ORDER) and
            split["split_codes"] == {"train": 0, "validation": 1, "test": 2} and
            features["feature_names"] == continual["feature_names"] and
            len(features["feature_names"]) == 54 and
            features["targets_excluded"] == ["Label", "ClassLabel"] and
            continual["scope"] == "continual" and continual["fit_rows"] == 6_347_232 and
            continual["reservoir_seed"] == 42):
        raise ValueError("Prepared data, class order, or frozen preprocessing changed")
    for name, expected in PREPARED_ARRAY_SHA256.items():
        if sha256_file(prepared / name) != expected:
            raise ValueError(f"Prepared input hash mismatch: {name}")
    if verify_full_partitions(large_data_paths(prepared), 9_167_581)["counts"] != split["counts"]:
        raise ValueError("Prepared split membership differs from saved counts")
    values = read_verified_acquisition(acquisition)
    evidence = verify_acquisition(values, split)
    train_counts = np.asarray([split["counts"][name]["train"] for name in BROAD_CLASS_ORDER],
                              dtype=np.int64)
    shift = prior_log_shift(train_counts, evidence["probabilities"])
    protocol = {"source_sha": source_sha,
        "study": "exploratory validation-only prediction-prior correction; no retraining",
        "historical_test_already_inspected": True,
        "acquisition_source_sha": ACQUISITION_SOURCE_SHA,
        "acquisition_zip_sha256": ZIP_SHA256,
        "prepared_input_sha256": INPUT_SHA256,
        "preprocessing_sha256": PREPROCESSING_SHA256,
        "class_ids": list(CLASS_IDS), "class_names": list(BROAD_CLASS_ORDER),
        "original_training_counts": train_counts.tolist(),
        "original_training_proportions": (train_counts / train_counts.sum()).tolist(),
        "recorded_sampling_probabilities": evidence["probabilities"].tolist(),
        "log_prior_shift": shift.tolist(),
        "rule": "argmax(logits + strength * log(original_training_prior / draw_prior))",
        "hypothesis": "approximate draw-prior correction may reduce Benign-to-Infiltration false positives; not guaranteed model calibration",
        "strength_grid": list(STRENGTHS), "zero_is_uncorrected_control": True,
        "acceptance_targets": TARGETS,
        "selection": "eligible only; highest full-validation macro-F1; ties lower Benign FPR, higher new mean recall, arm name, strength",
        "test_policy": "open fixed test only after an eligible validation selection lock; selected candidate once",
    }
    write_or_match(output / "protocol.json", protocol)
    preprocessor = FittedPreprocessor(
        tuple(continual["feature_names"]),
        np.asarray(continual["imputation_values"], dtype=np.float64),
        np.asarray(continual["means"], dtype=np.float64),
        np.asarray(continual["scales"], dtype=np.float64),
    )
    validation = DiskBackedFlowDataset(prepared, preprocessor, partition="validation",
                                       class_ids=CLASS_IDS)
    if len(validation) != 1_375_139:
        raise ValueError("Full validation split size changed")
    class_names = dict(enumerate(BROAD_CLASS_ORDER))
    rows = []
    for arm in ARMS:
        checkpoint_path = output / f"{arm}_verified_best.pt"
        if checkpoint_path.exists() and sha256_file(checkpoint_path) != EVIDENCE_SHA256[f"{arm}/best.pt"]:
            raise ValueError("Existing private checkpoint copy differs from verified evidence")
        if not checkpoint_path.exists():
            checkpoint_path.write_bytes(values[f"{arm}/best.pt"])
        config = json.loads(values[f"{arm}/arm_config.json"])
        saved_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        recorded_summary = json.loads(values[f"{arm}/arm_summary.json"])
        if not (saved_checkpoint["class_ids"] == list(CLASS_IDS) and
                saved_checkpoint["config"] == config["model"] and
                saved_checkpoint["data_identity"] == config["identity"] and
                saved_checkpoint["best_epoch"] == recorded_summary["best_epoch"] and
                saved_checkpoint["checkpoint_selection"] == "validation_macro_f1"):
            raise ValueError(f"{arm}: checkpoint metadata differs from verified run")
        del saved_checkpoint
        model = TabularTransformerClassifier.load(checkpoint_path, device="cuda")
        if not (model.class_ids == CLASS_IDS and model.config.num_features == 54 and
                model.config.num_classes == 8 and
                model.config.__dict__ == config["model"]):
            raise ValueError(f"{arm}: loaded checkpoint mapping/config differs")
        metrics_by_strength = evaluate_grid(model, validation, shift, STRENGTHS, class_names)
        historical = json.loads(values[f"{arm}/validation_metrics.json"])["metrics"]["combined_validation"]
        if metrics_by_strength[0.0]["confusion_matrix"] != historical["confusion_matrix"]:
            raise ValueError(f"{arm}: zero-strength validation differs from saved uncorrected result")
        for strength in STRENGTHS:
            metrics = metrics_by_strength[strength]
            rows.append({"arm": arm, "checkpoint_sha256": EVIDENCE_SHA256[f"{arm}/best.pt"],
                         "strength": strength, "summary": summarize_metrics(metrics),
                         "metrics": metrics})
        print(arm, "full validation scored at locked strengths", flush=True)
        del model
        torch.cuda.empty_cache()
    write_or_match(output / "validation_grid_metrics.json", {
        "partition": "full fixed validation", "rows": rows,
        "selection_used_test": False,
    })
    chosen = select_candidate(rows)
    lock = {"selected": None if chosen is None else
            {key: chosen[key] for key in ("arm", "strength", "checkpoint_sha256", "summary")},
            "eligibility_targets": TARGETS, "validation_only": True,
            "test_opened_before_lock": False,
            "eligible_count": sum(row["summary"]["eligible"] for row in rows)}
    write_or_match(output / "selection_lock.json", lock)
    if chosen is None:
        best_tradeoff = max(rows, key=lambda row: row["summary"]["macro_f1"])
        write_or_match(output / "best_validation_tradeoff_metrics.json", {"metrics": best_tradeoff["metrics"]})
        save_result_plots(output / "best_validation_tradeoff_metrics.json", output,
                          prefix="best_ineligible_validation")
        findings = {"status": "NO ELIGIBLE CANDIDATE; no test evaluation performed",
            "best_validation_macro_f1_tradeoff": {key: best_tradeoff[key]
                for key in ("arm", "strength", "summary")},
            "interpretation": "Prior correction did not meet all locked targets; do not claim successful calibration or eight-class deployment.",
            "historical_test_already_inspected": True}
        write_or_match(output / "findings.json", findings)
        print(findings["status"], flush=True)
        return
    # Only now open test rows, and score only the validation-selected pair.
    test = DiskBackedFlowDataset(prepared, preprocessor, partition="test", class_ids=CLASS_IDS)
    if len(test) != 1_375_134:
        raise ValueError("Fixed test split size changed")
    selected_checkpoint = output / f"{chosen['arm']}_verified_best.pt"
    selected_model = TabularTransformerClassifier.load(selected_checkpoint, device="cuda")
    test_metrics = evaluate_grid(selected_model, test, shift, (chosen["strength"],),
                                 class_names)[chosen["strength"]]
    result = {"arm": chosen["arm"], "strength": chosen["strength"],
        "checkpoint_sha256": chosen["checkpoint_sha256"],
        "selection_lock_sha256": sha256_file(output / "selection_lock.json"),
        "partition": "fixed full test, selected candidate only",
        "summary": summarize_metrics(test_metrics), "metrics": test_metrics}
    write_or_match(output / "selected_validation_metrics.json", {"metrics": chosen["metrics"]})
    write_or_match(output / "selected_test_metrics.json", result)
    for name in ("selected_validation", "selected_test"):
        save_result_plots(output / f"{name}_metrics.json", output, prefix=name)
    findings = {"status": "validation-eligible candidate selected; exploratory test result",
        "selected": lock["selected"], "selected_test_summary": result["summary"],
        "per_class_test": {str(i): {key: test_metrics["per_class"][str(i)][key]
            for key in ("precision", "recall", "f1", "support")} for i in CLASS_IDS},
        "historical_test_already_inspected": True,
        "interpretation": "Prediction-only adjustment; one validation-selected test pass does not establish broad robustness or probability calibration. Quarantine and training settings unchanged."}
    write_or_match(output / "findings.json", findings)
    print("Selected validation-eligible candidate:", chosen["arm"], chosen["strength"], flush=True)


if __name__ == "__main__":
    main()
