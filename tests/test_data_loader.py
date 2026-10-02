"""Synthetic smoke tests for Developer B's Parquet schema and batch interfaces."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.data import (
    ClassBalancedSample,
    DatasetSchemaError,
    PreparedDataset,
    SamplingError,
    inspect_parquet,
    iter_dataset_batches,
    materialize_selected_rows,
    prepare_sampled_dataset,
    sample_class_balanced_indices,
    stratified_split_indices,
)


class DataLoaderTests(unittest.TestCase):
    def _write_fixture(self, path: Path) -> None:
        table = pa.table(
            {
                "Flow Duration": pa.array([10, 20, 30], type=pa.int64()),
                "Fwd Header Length": pa.array([1, 2, 3], type=pa.int64()),
                "Bwd Header Length": pa.array([1, 2, 3], type=pa.int64()),
                "Fwd Seg Size Min": pa.array([1, 2, 3], type=pa.int32()),
                "Packet Length Mean": pa.array([1.0, 2.0, 3.0], type=pa.float32()),
                "Label": pa.array(["fine-a", "fine-b", "fine-a"]),
                "ClassLabel": pa.array(["class-a", "class-b", "class-a"]),
            }
        )
        pq.write_table(table, path)

    def test_inspection_retains_only_audited_features(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.parquet"
            self._write_fixture(path)

            info = inspect_parquet(path, strict_schema=False)

            self.assertEqual(info.row_count, 3)
            self.assertEqual(
                info.schema.raw_numeric_features,
                (
                    "Flow Duration",
                    "Fwd Header Length",
                    "Bwd Header Length",
                    "Fwd Seg Size Min",
                    "Packet Length Mean",
                ),
            )
            self.assertEqual(
                info.schema.retained_feature_names,
                ("Flow Duration", "Packet Length Mean"),
            )

    def test_batch_loader_keeps_memory_bounded_and_columns_stable(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.parquet"
            self._write_fixture(path)

            batches = list(iter_dataset_batches(path, batch_size=2, strict_schema=False))

            self.assertEqual([batch.num_rows for batch in batches], [2, 1])
            self.assertEqual(
                batches[0].schema.names,
                ["Flow Duration", "Packet Length Mean", "Label", "ClassLabel"],
            )

    def test_missing_target_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.parquet"
            pq.write_table(pa.table({"Flow Duration": [1, 2]}), path)

            with self.assertRaises(DatasetSchemaError):
                inspect_parquet(path, strict_schema=False)

    def test_reservoir_sample_is_balanced_and_seeded(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sample.parquet"
            table = pa.table(
                {
                    "Flow Duration": list(range(20)),
                    "Label": ["fine-a"] * 10 + ["fine-b"] * 10,
                    "ClassLabel": ["class-a"] * 10 + ["class-b"] * 10,
                }
            )
            pq.write_table(table, path)

            first = sample_class_balanced_indices(
                path,
                classes=("class-a", "class-b"),
                per_class_limit=4,
                seed=11,
                strict_schema=False,
            )
            second = sample_class_balanced_indices(
                path,
                classes=("class-a", "class-b"),
                per_class_limit=4,
                seed=11,
                strict_schema=False,
            )

            self.assertTrue((first.row_indices == second.row_indices).all())
            self.assertEqual(first.labels.tolist().count("class-a"), 4)
            self.assertEqual(first.labels.tolist().count("class-b"), 4)
            self.assertEqual(first.available_counts, {"class-a": 10, "class-b": 10})

    def test_reservoir_sample_rejects_an_unavailable_class_size(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sample.parquet"
            pq.write_table(
                pa.table(
                    {
                        "Flow Duration": [1, 2],
                        "Label": ["fine-a", "fine-a"],
                        "ClassLabel": ["class-a", "class-a"],
                    }
                ),
                path,
            )

            with self.assertRaises(SamplingError):
                sample_class_balanced_indices(
                    path,
                    classes=("class-a",),
                    per_class_limit=3,
                    seed=11,
                    strict_schema=False,
                )

    def test_stratified_split_is_complete_disjoint_and_repeatable(self) -> None:
        labels = np.asarray(["a"] * 20 + ["b"] * 20, dtype=object)
        first = stratified_split_indices(labels, seed=17)
        second = stratified_split_indices(labels, seed=17)

        self.assertEqual((first.train.size, first.validation.size, first.test.size), (28, 6, 6))
        self.assertTrue((first.train == second.train).all())
        all_positions = np.concatenate((first.train, first.validation, first.test))
        self.assertEqual(len(set(all_positions.tolist())), labels.size)
        for partition in (first.train, first.validation, first.test):
            self.assertEqual((labels[partition] == "a").sum(), partition.size // 2)
            self.assertEqual((labels[partition] == "b").sum(), partition.size // 2)

    def test_preprocessing_materializes_rows_in_requested_order_and_fits_train_only(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "preprocessing.parquet"
            labels = np.asarray(["class-a"] * 10 + ["class-b"] * 10, dtype=object)
            table = pa.table(
                {
                    "Flow Duration": [-1] + list(range(1, 20)),
                    "Fwd Header Length": [1] * 20,
                    "Bwd Header Length": [1] * 20,
                    "Fwd Seg Size Min": [1] * 20,
                    "Init Fwd Win Bytes": [-1 if index % 2 == 0 else 100 for index in range(20)],
                    "Packet Length Mean": list(range(20)),
                    "Label": ["fine-a"] * 10 + ["fine-b"] * 10,
                    "ClassLabel": labels.tolist(),
                }
            )
            pq.write_table(table, path)

            selected = materialize_selected_rows(
                path,
                np.asarray([12, 1, 9]),
                strict_schema=False,
            )
            self.assertEqual(selected.labels.tolist(), ["class-b", "class-a", "class-a"])

            sample = ClassBalancedSample(
                row_indices=np.arange(20, dtype=np.int64),
                labels=labels,
                class_order=("class-a", "class-b"),
                available_counts={"class-a": 10, "class-b": 10},
                per_class_limit=10,
                seed=0,
            )
            split = stratified_split_indices(sample.labels, seed=7)
            prepared = prepare_sampled_dataset(path, sample, split, strict_schema=False)

            self.assertIsInstance(prepared, PreparedDataset)
            self.assertEqual(prepared.feature_names, ("Flow Duration", "Init Fwd Win Bytes", "Packet Length Mean"))
            self.assertEqual(prepared.train.features.dtype, np.float32)
            self.assertTrue(np.isfinite(prepared.train.features).all())
            self.assertTrue(
                np.allclose(prepared.train.features.mean(axis=0), 0.0, atol=1e-6)
            )
            self.assertEqual(prepared.class_names, {0: "class-a", 1: "class-b"})


if __name__ == "__main__":
    unittest.main()
