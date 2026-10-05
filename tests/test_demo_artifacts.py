import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from src.demo.artifacts import (
    ArtifactConfigurationError,
    ModelArtifact,
    load_demo_config,
    load_fitted_preprocessor,
)
from src.demo.inference import UploadSchemaError, prepare_upload


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DemoArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / "model.pt"
        self.checkpoint.write_bytes(b"checkpoint-placeholder")
        self.static = self.root / "static.json"
        self.continual = self.root / "continual.json"
        for path, scope in ((self.static, "static"), (self.continual, "continual")):
            path.write_text(
                json.dumps(
                    {
                        "scope": scope,
                        "feature_names": ["first", "second"],
                        "imputation_values": [0.0, 1.0],
                        "means": [1.0, 2.0],
                        "scales": [2.0, 4.0],
                    }
                )
            )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _model(self, scope: str = "static") -> ModelArtifact:
        preprocessor = self.static if scope == "static" else self.continual
        return ModelArtifact(
            key="model",
            display_name="Test model",
            identity="test:model",
            checkpoint=self.checkpoint,
            checkpoint_sha256=_sha(self.checkpoint),
            preprocessor=preprocessor,
            preprocessor_sha256=_sha(preprocessor),
            preprocessing_scope=scope,
            class_names={0: "Benign", 1: "Attack"},
        )

    def test_config_resolves_paths_and_separates_preprocessing_scopes(self) -> None:
        archive = self.root / "results.zip"
        replay = self.root / "replay.npz"
        archive.write_bytes(b"zip-placeholder")
        replay.write_bytes(b"npz-placeholder")
        models = {}
        for key, scope, names in (
            ("clean", "static", [str(index) for index in range(8)]),
            ("teacher", "continual", [str(index) for index in range(5)]),
        ):
            preprocessor = self.static if scope == "static" else self.continual
            models[key] = {
                "display_name": key,
                "identity": key + ":identity",
                "checkpoint": self.checkpoint.name,
                "checkpoint_sha256": _sha(self.checkpoint),
                "preprocessor": preprocessor.name,
                "preprocessor_sha256": _sha(preprocessor),
                "preprocessing_scope": scope,
                "class_names": names,
            }
        config_path = self.root / "demo.json"
        config_path.write_text(
            json.dumps(
                {
                    "base_dir": ".",
                    "max_inference_rows": 10,
                    "default_model": "clean",
                    "task1_teacher": "teacher",
                    "models": models,
                    "replay_buffer": replay.name,
                    "task2_results_zip": archive.name,
                    "task2_results_sha256": _sha(archive),
                    "quarantine_db": "queue/review.sqlite3",
                }
            )
        )
        config = load_demo_config(config_path)
        self.assertEqual(config.models["clean"].preprocessing_scope, "static")
        self.assertEqual(config.models["teacher"].preprocessing_scope, "continual")
        self.assertEqual(config.quarantine_db, self.root / "queue/review.sqlite3")

    def test_preprocessor_rejects_wrong_scope(self) -> None:
        model = self._model("static")
        wrong = ModelArtifact(
            **{**model.__dict__, "preprocessing_scope": "continual"}
        )
        with self.assertRaisesRegex(ArtifactConfigurationError, "requires 'continual'"):
            load_fitted_preprocessor(wrong)

    def test_upload_requires_exact_feature_order_and_encodes_optional_labels(self) -> None:
        model = self._model()
        state = load_fitted_preprocessor(model)
        frame = pd.DataFrame(
            {"first": [1.0, 3.0], "second": [2.0, 6.0], "ClassLabel": ["Benign", "Attack"]}
        )
        prepared = prepare_upload(
            frame, model, state, max_rows=2, label_column="ClassLabel"
        )
        np.testing.assert_array_equal(prepared.labels, np.asarray([0, 1]))
        np.testing.assert_allclose(
            prepared.transformed_features, np.asarray([[0, 0], [1, 1]], dtype=np.float32)
        )
        reordered = frame[["second", "first", "ClassLabel"]]
        with self.assertRaisesRegex(UploadSchemaError, "order differs"):
            prepare_upload(
                reordered, model, state, max_rows=2, label_column="ClassLabel"
            )

    def test_upload_bound_is_enforced_before_inference(self) -> None:
        model = self._model()
        state = load_fitted_preprocessor(model)
        frame = pd.DataFrame({"first": [1.0, 2.0], "second": [3.0, 4.0]})
        with self.assertRaisesRegex(UploadSchemaError, "1 to 1 rows"):
            prepare_upload(frame, model, state, max_rows=1)


if __name__ == "__main__":
    unittest.main()
