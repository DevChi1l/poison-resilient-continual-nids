import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

import numpy as np

from src.demo.artifacts import ArtifactConfigurationError
from src.demo.results import ARM_ORDER, load_task2_playback, normalized_confusion


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DemoResultPlaybackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _archive(self, *, unsafe: bool = False) -> Path:
        path = self.root / "results.zip"
        prefix = "run/"
        metric = {
            "accuracy": 0.5,
            "macro_f1": 0.4,
            "benign_false_positive": {"rate": 0.2},
            "class_ids": [0, 1],
            "per_class": {"0": {"name": "Benign"}, "1": {"name": "Attack"}},
            "confusion_matrix": [[3, 1], [2, 2]],
        }
        with ZipFile(path, "w") as archive:
            archive.writestr(
                prefix + "comparison.json",
                json.dumps({"conditions": list(ARM_ORDER), "arms": {}}),
            )
            archive.writestr(prefix + "calibration.json", json.dumps({"threshold": 0.1}))
            archive.writestr(prefix + "gate_audit.json", json.dumps({"audit": {}}))
            for arm in ARM_ORDER:
                archive.writestr(
                    f"{prefix}{arm}/combined_test_metrics.json", json.dumps(metric)
                )
            if unsafe:
                archive.writestr("..\\escape.txt", "not extracted")
        return path

    def test_reads_only_named_saved_evidence_and_normalizes_confusion(self) -> None:
        path = self._archive()
        playback = load_task2_playback(path, _sha(path))
        self.assertEqual(playback.run_prefix, "run/")
        normalized = normalized_confusion(playback.combined_metrics[ARM_ORDER[0]])
        np.testing.assert_allclose(normalized, [[0.75, 0.25], [0.5, 0.5]])

    def test_rejects_archive_with_path_traversal_member(self) -> None:
        path = self._archive(unsafe=True)
        with self.assertRaisesRegex(ArtifactConfigurationError, "Unsafe ZIP member"):
            load_task2_playback(path, _sha(path))


if __name__ == "__main__":
    unittest.main()
