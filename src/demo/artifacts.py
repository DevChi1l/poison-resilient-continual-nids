"""Configuration and integrity checks for private local demo artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.data.preprocessing import FittedPreprocessor


class ArtifactConfigurationError(ValueError):
    """Raised when artifact paths, identities, or pairings are inconsistent."""


@dataclass(frozen=True)
class ModelArtifact:
    """One checkpoint and its required inference-time contract."""

    key: str
    display_name: str
    identity: str
    checkpoint: Path
    checkpoint_sha256: str
    preprocessor: Path
    preprocessor_sha256: str
    preprocessing_scope: str
    class_names: dict[int, str]
    novelty_artifact: Path | None = None
    novelty_label: str | None = None

    @property
    def class_ids(self) -> tuple[int, ...]:
        return tuple(self.class_names)


@dataclass(frozen=True)
class DemoArtifactConfig:
    """Resolved paths and bounds used by the three local UI views."""

    source_path: Path
    max_inference_rows: int
    default_model: str
    models: dict[str, ModelArtifact]
    task1_teacher: str
    replay_buffer: Path
    task2_results_zip: Path
    task2_results_sha256: str
    quarantine_db: Path


def file_sha256(path: str | Path) -> str:
    """Hash a local file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(record: Mapping[str, Any], name: str) -> str:
    value = record.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ArtifactConfigurationError(f"{name} must be a non-empty string")
    return value


def _resolve(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _class_names(value: Any, model_key: str) -> dict[int, str]:
    if isinstance(value, list):
        names = {index: name for index, name in enumerate(value)}
    elif isinstance(value, dict):
        try:
            names = {int(class_id): name for class_id, name in value.items()}
        except (TypeError, ValueError) as error:
            raise ArtifactConfigurationError(
                f"models.{model_key}.class_names keys must be integer IDs"
            ) from error
    else:
        raise ArtifactConfigurationError(
            f"models.{model_key}.class_names must be a list or object"
        )
    if not names or any(not isinstance(name, str) or not name for name in names.values()):
        raise ArtifactConfigurationError(
            f"models.{model_key}.class_names must contain non-empty names"
        )
    if tuple(names) != tuple(range(len(names))):
        raise ArtifactConfigurationError(
            f"models.{model_key}.class_names must use ordered contiguous IDs from zero"
        )
    return names


def load_demo_config(path: str | Path) -> DemoArtifactConfig:
    """Load JSON configuration and resolve all paths without opening artifacts."""

    source = Path(path).expanduser().resolve()
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ArtifactConfigurationError(f"Demo config does not exist: {source}") from error
    except json.JSONDecodeError as error:
        raise ArtifactConfigurationError(f"Demo config is not valid JSON: {error}") from error
    if not isinstance(record, dict):
        raise ArtifactConfigurationError("Demo config root must be an object")

    base_value = record.get("base_dir", ".")
    if not isinstance(base_value, str):
        raise ArtifactConfigurationError("base_dir must be a path string")
    base_dir = _resolve(source.parent, base_value)

    raw_models = record.get("models")
    if not isinstance(raw_models, dict) or not raw_models:
        raise ArtifactConfigurationError("models must be a non-empty object")
    models: dict[str, ModelArtifact] = {}
    for key, raw in raw_models.items():
        if not isinstance(key, str) or not isinstance(raw, dict):
            raise ArtifactConfigurationError("each models entry must be an object")
        novelty_path = raw.get("novelty_artifact")
        models[key] = ModelArtifact(
            key=key,
            display_name=_required_text(raw, "display_name"),
            identity=_required_text(raw, "identity"),
            checkpoint=_resolve(base_dir, _required_text(raw, "checkpoint")),
            checkpoint_sha256=_required_text(raw, "checkpoint_sha256").lower(),
            preprocessor=_resolve(base_dir, _required_text(raw, "preprocessor")),
            preprocessor_sha256=_required_text(raw, "preprocessor_sha256").lower(),
            preprocessing_scope=_required_text(raw, "preprocessing_scope"),
            class_names=_class_names(raw.get("class_names"), key),
            novelty_artifact=(
                _resolve(base_dir, novelty_path)
                if isinstance(novelty_path, str) and novelty_path
                else None
            ),
            novelty_label=(
                str(raw["novelty_label"])
                if raw.get("novelty_label") is not None
                else None
            ),
        )

    max_rows = record.get("max_inference_rows", 2_000)
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
        raise ArtifactConfigurationError("max_inference_rows must be a positive integer")
    default_model = _required_text(record, "default_model")
    teacher = _required_text(record, "task1_teacher")
    if default_model not in models:
        raise ArtifactConfigurationError("default_model is absent from models")
    if teacher not in models:
        raise ArtifactConfigurationError("task1_teacher is absent from models")
    if len(models[teacher].class_names) != 5:
        raise ArtifactConfigurationError("task1_teacher must identify the five-class teacher")

    return DemoArtifactConfig(
        source_path=source,
        max_inference_rows=max_rows,
        default_model=default_model,
        models=models,
        task1_teacher=teacher,
        replay_buffer=_resolve(base_dir, _required_text(record, "replay_buffer")),
        task2_results_zip=_resolve(base_dir, _required_text(record, "task2_results_zip")),
        task2_results_sha256=_required_text(record, "task2_results_sha256").lower(),
        quarantine_db=_resolve(base_dir, _required_text(record, "quarantine_db")),
    )


def verify_file(path: Path, expected_sha256: str, *, label: str) -> str:
    """Require a file and match its declared SHA-256."""

    if not path.is_file():
        raise ArtifactConfigurationError(f"{label} does not exist: {path}")
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ArtifactConfigurationError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, found {actual}"
        )
    return actual


def load_fitted_preprocessor(model: ModelArtifact) -> FittedPreprocessor:
    """Load and validate the preprocessor paired with a configured model."""

    verify_file(
        model.preprocessor,
        model.preprocessor_sha256,
        label=f"{model.display_name} preprocessor",
    )
    try:
        record = json.loads(model.preprocessor.read_text(encoding="utf-8"))
        scope = record["scope"]
        feature_names = tuple(record["feature_names"])
        imputation = np.asarray(record["imputation_values"], dtype=np.float64)
        means = np.asarray(record["means"], dtype=np.float64)
        scales = np.asarray(record["scales"], dtype=np.float64)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ArtifactConfigurationError(
            f"Invalid preprocessing artifact {model.preprocessor}: {error}"
        ) from error
    if scope != model.preprocessing_scope:
        raise ArtifactConfigurationError(
            f"{model.display_name} requires {model.preprocessing_scope!r} preprocessing, "
            f"but artifact declares {scope!r}"
        )
    width = len(feature_names)
    if width == 0 or any(values.shape != (width,) for values in (imputation, means, scales)):
        raise ArtifactConfigurationError("Preprocessing vectors do not match feature_names")
    if not all(np.isfinite(values).all() for values in (imputation, means, scales)):
        raise ArtifactConfigurationError("Preprocessing vectors contain non-finite values")
    if np.any(scales <= 0):
        raise ArtifactConfigurationError("Preprocessing scales must be positive")
    return FittedPreprocessor(feature_names, imputation, means, scales)


def load_task1_novelty_threshold(model: ModelArtifact) -> float:
    """Load the weak max-confidence cutoff only for a five-class Task 1 model."""

    if len(model.class_ids) != 5 or model.novelty_artifact is None:
        raise ArtifactConfigurationError(
            "A novelty cutoff is valid here only for the configured five-class Task 1 model"
        )
    try:
        record = json.loads(model.novelty_artifact.read_text(encoding="utf-8"))
        threshold = float(record["calibration"]["threshold"])
        head_ids = tuple(int(value) for value in record["class_ids_in_task1_head"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ArtifactConfigurationError(f"Invalid Task 1 novelty artifact: {error}") from error
    if head_ids != model.class_ids or not 0 <= threshold <= 1:
        raise ArtifactConfigurationError("Task 1 novelty artifact does not match the model head")
    return threshold
