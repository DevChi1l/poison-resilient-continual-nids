import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.schema import DROPPED_FEATURES
from src.demo.artifacts import DatasetArtifact, DemoArtifactConfig, ModelArtifact
from src.demo.sample_export import SampleExportError, export_rehearsal_sample


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DemoSampleExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.class_names = (
            "Benign",
            "DDoS",
            "DoS",
            "Botnet",
            "Bruteforce",
            "Infiltration",
            "Webattack",
            "Portscan",
        )
        self.feature_names = tuple(f"Feature {index}" for index in range(54))
        row_count = 48
        labels = np.repeat(self.class_names, 6)
        columns = {
            name: np.arange(row_count, dtype=np.float64) + index
            for index, name in enumerate(self.feature_names)
        }
        for name in DROPPED_FEATURES:
            columns[name] = np.arange(row_count, dtype=np.int64)
        columns["Label"] = ["fine"] * row_count
        columns["ClassLabel"] = labels.tolist()
        self.parquet = self.root / "source.parquet"
        pq.write_table(pa.table(columns), self.parquet)

        self.split_codes = self.root / "splits.npy"
        self.test_indices = self.root / "test_indices.npy"
        np.save(self.split_codes, np.full(row_count, 2, dtype=np.uint8))
        np.save(self.test_indices, np.arange(row_count, dtype=np.int64))
        self.split_manifest = self.root / "split_manifest.json"
        self.split_manifest.write_text(
            json.dumps(
                {
                    "row_count": row_count,
                    "class_order": list(self.class_names),
                    "counts": {
                        name: {"train": 0, "validation": 0, "test": 6}
                        for name in self.class_names
                    },
                }
            )
        )
        self.preprocessor = self.root / "preprocessing_static.json"
        self.preprocessor.write_text(
            json.dumps(
                {
                    "scope": "static",
                    "feature_names": list(self.feature_names),
                    "imputation_values": [0.0] * 54,
                    "means": [0.0] * 54,
                    "scales": [1.0] * 54,
                }
            )
        )
        checkpoint = self.root / "best.pt"
        checkpoint.write_bytes(b"not loaded by sample export")
        model = ModelArtifact(
            key="clean",
            display_name="clean",
            identity="clean:test",
            checkpoint=checkpoint,
            checkpoint_sha256=_sha(checkpoint),
            preprocessor=self.preprocessor,
            preprocessor_sha256=_sha(self.preprocessor),
            preprocessing_scope="static",
            class_names=dict(enumerate(self.class_names)),
        )
        dataset = DatasetArtifact(
            source_parquet=self.parquet,
            source_sha256=_sha(self.parquet),
            split_manifest=self.split_manifest,
            split_manifest_sha256=_sha(self.split_manifest),
            split_codes=self.split_codes,
            split_codes_sha256=_sha(self.split_codes),
            test_indices=self.test_indices,
            test_indices_sha256=_sha(self.test_indices),
        )
        self.config = DemoArtifactConfig(
            source_path=self.root / "config.json",
            max_inference_rows=100,
            default_model="clean",
            models={"clean": model},
            task1_teacher="clean",
            replay_buffer=self.root / "replay.npz",
            replay_buffer_sha256="unused",
            task2_results_zip=self.root / "results.zip",
            task2_results_sha256="unused",
            quarantine_db=self.root / "queue.sqlite3",
            dataset=dataset,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_exports_exact_feature_order_all_classes_and_separate_ids(self) -> None:
        first = export_rehearsal_sample(
            self.config, self.root / "first", rows_per_class=2, seed=42, batch_size=7
        )
        second = export_rehearsal_sample(
            self.config, self.root / "second", rows_per_class=2, seed=42, batch_size=11
        )
        flow = pd.read_csv(first.flow_csv)
        provenance = pd.read_csv(first.provenance_csv)
        self.assertEqual(flow.columns.tolist(), [*self.feature_names, "ClassLabel"])
        self.assertNotIn("source_row_id", flow.columns)
        self.assertEqual(provenance.columns.tolist(), ["sample_row", "source_row_id", "ClassLabel"])
        self.assertEqual(flow["ClassLabel"].value_counts().to_dict(), {name: 2 for name in self.class_names})
        np.testing.assert_array_equal(first.source_row_ids, second.source_row_ids)
        manifest = json.loads(first.manifest_json.read_text())
        self.assertTrue(manifest["selection"]["prediction_independent"])
        self.assertIn("not full-test performance", manifest["limitations"][1])

    def test_refuses_to_overwrite_generated_flows(self) -> None:
        output = self.root / "sample"
        export_rehearsal_sample(self.config, output, rows_per_class=1)
        with self.assertRaisesRegex(SampleExportError, "Refusing to overwrite"):
            export_rehearsal_sample(self.config, output, rows_per_class=1)


if __name__ == "__main__":
    unittest.main()
