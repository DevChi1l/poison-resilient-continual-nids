"""Synthetic checks for plots derived from saved confusion counts."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.evaluation.plots import confusion_plot_data, save_result_plots


class SavedPlotTests(unittest.TestCase):
    def setUp(self):
        self.record = {"metrics": {"class_ids": [0, 5, 7],
                                   "confusion_matrix": [[8, 2, 0], [1, 1, 0], [0, 0, 0]],
                                   "per_class": {"0": {"name": "Benign"},
                                                 "5": {"name": "Infiltration"},
                                                 "7": {"name": "Portscan"}}}}

    def test_external_ids_row_normalization_and_zero_support(self):
        original = json.dumps(self.record)
        data = confusion_plot_data(self.record)
        np.testing.assert_allclose(data["normalized"],
                                   [[0.8, 0.2, 0], [0.5, 0.5, 0], [0, 0, 0]])
        np.testing.assert_allclose(data["recall"], [0.8, 0.5, 0])
        np.testing.assert_allclose(data["f1"], [16 / 19, 0.4, 0])
        self.assertEqual(data["class_names"], ["Benign", "Infiltration", "Portscan"])
        self.assertEqual(json.dumps(self.record), original)

    def test_saved_json_generates_distinct_plots_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "test_metrics.json"
            source.write_text(json.dumps(self.record))
            outputs = save_result_plots(source, root / "plots", prefix="saved")
            self.assertEqual(len(outputs), 3)
            self.assertTrue(all(Path(path).stat().st_size > 0 for path in outputs.values()))
            self.assertEqual(json.loads(source.read_text()), self.record)
            with self.assertRaises(FileExistsError):
                save_result_plots(source, root / "plots", prefix="saved")

    def test_invalid_counts_rejected(self):
        for counts in ([[1, -1], [0, 1]], [[1.5, 0], [0, 1]], [[1, 0, 0]]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                confusion_plot_data({"class_ids": [0, 1], "confusion_matrix": counts})


if __name__ == "__main__":
    unittest.main()
