"""Read-only preflight checks for the artifact-backed local demonstration."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sqlite3

import numpy as np

from src.data.loader import inspect_parquet

from .artifacts import (
    ArtifactConfigurationError,
    DemoArtifactConfig,
    load_demo_config,
    load_task1_novelty_threshold,
    verify_file,
)
from .inference import load_verified_model
from .quarantine import load_replay_buffer
from .results import ARM_ORDER, load_task2_playback, normalized_confusion
from .sample_export import load_saved_test_indices


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    passed: bool
    detail: str
    fix: str | None = None


@dataclass(frozen=True)
class PreflightReport:
    config: Path
    database: Path | None
    checks: tuple[PreflightCheck, ...]

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)


def _pass(name: str, detail: str) -> PreflightCheck:
    return PreflightCheck(name, True, detail)


def _fail(name: str, error: Exception | str, fix: str) -> PreflightCheck:
    return PreflightCheck(name, False, str(error), fix)


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.parent
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _check_sqlite_location(path: Path) -> PreflightCheck:
    name = "SQLite review queue"
    try:
        if path.exists():
            if not path.is_file():
                raise ValueError(f"configured queue path is not a file: {path}")
            uri = f"file:{path.resolve().as_posix()}?mode=ro"
            with sqlite3.connect(uri, uri=True) as connection:
                result = connection.execute("PRAGMA quick_check").fetchone()
                if result is None or result[0] != "ok":
                    raise ValueError(f"SQLite quick_check returned {result}")
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
            expected = {"review_items", "decision_history"}
            if not expected.issubset(tables):
                raise ValueError("existing queue lacks the review/audit tables")
            return _pass(name, f"readable existing queue: {path}")
        parent = _nearest_existing_parent(path)
        if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
            raise PermissionError(f"nearest existing parent is not writable: {parent}")
        return _pass(name, f"new queue can be created under writable parent {parent}")
    except (OSError, sqlite3.Error, ValueError) as error:
        return _fail(
            name,
            error,
            "choose --database under a writable local directory; do not point it at a prior demo queue",
        )


def run_preflight(
    config_path: str | Path,
    *,
    database_override: str | Path | None = None,
) -> PreflightReport:
    """Run bounded, read-only compatibility checks; never train a model."""

    checks: list[PreflightCheck] = []
    try:
        config = load_demo_config(config_path)
    except ArtifactConfigurationError as error:
        return PreflightReport(
            Path(config_path).expanduser().resolve(),
            None,
            (
                _fail(
                    "artifact configuration",
                    error,
                    "copy configs/demo_artifacts.example.json and correct its base_dir and artifact paths",
                ),
            ),
        )
    checks.append(_pass("artifact configuration", f"loaded {config.source_path}"))

    loaded_models = {}
    for key, spec in config.models.items():
        try:
            classifier, state = load_verified_model(spec)
            loaded_models[key] = (classifier, state)
            checks.append(
                _pass(
                    f"model {key}",
                    f"hashes and compatibility verified: {classifier.config.num_features} features, "
                    f"{len(classifier.class_ids)} classes, scope={spec.preprocessing_scope}",
                )
            )
        except (ArtifactConfigurationError, ImportError, RuntimeError, OSError) as error:
            checks.append(
                _fail(
                    f"model {key}",
                    error,
                    "restore the configured checkpoint/preprocessor pair or install the CPU demo dependencies",
                )
            )

    teacher = config.models[config.task1_teacher]
    try:
        threshold = load_task1_novelty_threshold(teacher)
        checks.append(
            _pass(
                "Task 1 novelty artifact",
                f"hash verified; weak experimental threshold={threshold:.10f}",
            )
        )
    except (ArtifactConfigurationError, OSError) as error:
        checks.append(
            _fail(
                "Task 1 novelty artifact",
                error,
                "restore the configured Task 1 novelty JSON; never substitute the eight-class model",
            )
        )

    try:
        verify_file(
            config.replay_buffer,
            config.replay_buffer_sha256,
            label="Task 1 replay buffer",
        )
        replay = load_replay_buffer(config.replay_buffer)
        teacher_state = loaded_models.get(config.task1_teacher, (None, None))[1]
        if teacher_state is None:
            raise ValueError("teacher compatibility failed earlier")
        if replay.features.shape != (500, len(teacher_state.feature_names)):
            raise ValueError(
                f"expected replay shape (500, {len(teacher_state.feature_names)}), "
                f"found {replay.features.shape}"
            )
        if not np.isin(replay.labels, teacher.class_ids).all():
            raise ValueError("replay labels are outside the Task 1 teacher head")
        checks.append(
            _pass(
                "Task 1 replay buffer",
                f"hash verified; features={replay.features.shape}, labels={replay.labels.shape}",
            )
        )
    except (ArtifactConfigurationError, OSError, ValueError) as error:
        checks.append(
            _fail(
                "Task 1 replay buffer",
                error,
                "restore the configured 500x54 replay NPZ that belongs to the Task 1 teacher",
            )
        )

    try:
        playback = load_task2_playback(
            config.task2_results_zip, config.task2_results_sha256
        )
        for arm in ARM_ORDER:
            matrix = normalized_confusion(playback.combined_metrics[arm])
            if matrix.shape != (8, 8):
                raise ArtifactConfigurationError(
                    f"{arm} confusion matrix has shape {matrix.shape}, expected (8, 8)"
                )
        audit = playback.gate_audit["audit"]
        if (
            int(audit["poison_rejected"]) != 80
            or int(audit["poisoned_candidates"]) != 80
            or int(audit["clean_false_rejected"]) != 42
            or int(audit["clean_candidates"]) != 420
            or int(audit["retained_total"]) != 378
        ):
            raise ArtifactConfigurationError("Task 2 gate counts differ from recorded evidence")
        checks.append(
            _pass(
                "Task 2 saved evidence",
                f"hash verified and named JSON readable for {', '.join(ARM_ORDER)}; gate=122 rejected",
            )
        )
    except (ArtifactConfigurationError, KeyError, TypeError, ValueError, OSError) as error:
        checks.append(
            _fail(
                "Task 2 saved evidence",
                error,
                "restore the original results ZIP matching task2_results_sha256; do not repackage it",
            )
        )

    if config.dataset is None:
        checks.append(
            _fail(
                "held-out dataset provenance",
                "dataset section is absent",
                "add the dataset section from configs/demo_artifacts.example.json",
            )
        )
    else:
        dataset = config.dataset
        try:
            verify_file(dataset.source_parquet, dataset.source_sha256, label="source Parquet")
            verify_file(
                dataset.split_manifest,
                dataset.split_manifest_sha256,
                label="split manifest",
            )
            verify_file(dataset.split_codes, dataset.split_codes_sha256, label="split-code array")
            verify_file(dataset.test_indices, dataset.test_indices_sha256, label="test-index array")
            info = inspect_parquet(dataset.source_parquet, strict_schema=True)
            manifest = json.loads(dataset.split_manifest.read_text(encoding="utf-8"))
            if manifest.get("row_count") != info.row_count:
                raise ValueError("split manifest and Parquet row counts differ")
            static_state = loaded_models.get(config.default_model, (None, None))[1]
            if static_state is None:
                raise ValueError("static model compatibility failed earlier")
            if tuple(info.schema.retained_feature_names) != static_state.feature_names:
                raise ValueError("Parquet feature order differs from static preprocessor")
            indices = load_saved_test_indices(
                dataset.test_indices, row_count=info.row_count
            )
            splits = np.load(dataset.split_codes, mmap_mode="r", allow_pickle=False)
            if splits.dtype != np.dtype("uint8") or splits.shape != (info.row_count,):
                raise ValueError("split-code array must be row-aligned uint8")
            test_count = int(np.count_nonzero(splits == 2))
            if test_count != len(indices):
                raise ValueError("test indices and split-code test count differ")
            for start in range(0, len(indices), 262_144):
                if np.any(splits[indices[start : start + 262_144]] != 2):
                    raise ValueError("a saved test index does not have test split code 2")
            checks.append(
                _pass(
                    "held-out dataset provenance",
                    f"all hashes verified; rows={info.row_count:,}, saved test rows={len(indices):,}, features=54",
                )
            )
        except (
            ArtifactConfigurationError,
            FileNotFoundError,
            ImportError,
            json.JSONDecodeError,
            OSError,
            ValueError,
        ) as error:
            checks.append(
                _fail(
                    "held-out dataset provenance",
                    error,
                    "restore the original Parquet and saved split arrays; if only arrays are absent, rerun "
                    "prepare_full_partitions with seed 42 and require the configured hashes before export",
                )
            )

    database = (
        Path(database_override).expanduser().resolve()
        if database_override is not None
        else config.quarantine_db
    )
    checks.append(_check_sqlite_location(database))
    return PreflightReport(config.source_path, database, tuple(checks))
